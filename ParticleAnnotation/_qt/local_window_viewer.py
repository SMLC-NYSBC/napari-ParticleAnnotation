import warnings

import scipy.ndimage as nd

import torch
from magicgui.widgets import (
    Container,
    HBox,
    VBox,
    SpinBox,
    create_widget,
    PushButton,
    LineEdit,
    FloatSlider,
)

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter

from napari import Viewer
from scipy.spatial import KDTree
from topaz.stats import normalize
from vispy.geometry import Rect

import numpy as np
from qtpy.QtWidgets import QFileDialog

from scipy.ndimage import maximum_filter
from napari.layers import Points
from napari.utils.notifications import show_info
import napari
from ParticleAnnotation.utils.model.active_learning_model import (
    BinaryLogisticRegression,
    initialize_model,
    label_points_to_mask,
    predict_3d_with_AL,
    update_true_labels,
)

from ParticleAnnotation.utils.load_data import (
    downsample,
    load_template,
    load_coordinates,
    save_coordinates,
)
from ParticleAnnotation.utils.model.utils import (
    rank_candidate_locations,
    get_device,
    get_random_patch,
    correct_coord,
    find_peaks,
)
from ParticleAnnotation._qt.viewer_utils import (
    ViewerModel,
    QtViewerWrap,
    get_property_names,
    copy_layer,
    copy_layer_viewer2,
    OwnPartial,
)


class MultipleViewerWidget(QSplitter):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = ViewerModel(title="View_1")
        self.viewer_model2 = ViewerModel(title="View_2")
        self._block = False
        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(viewer, self.viewer_model2)

        self.points_layer = self.viewer_model1.add_points(
            [0, 0, 0], name="Mouse Pointer", symbol="cross", size=2
        )
        self.annotation_widget = AnnotationWidgetv2(viewer)
        viewer.window.add_dock_widget(
            self.annotation_widget, name="Annotation", area="left"
        )

        # self.annotation_widget.reset_view.clicked.connect(self.reset_view)

        self.viewer.camera.events.connect(self._sync_view)

        # Create new viewers
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        viewer_splitter.addWidget(self.qt_viewer1)
        viewer_splitter.addWidget(self.qt_viewer2)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        self.addWidget(viewer_splitter)

        # Add/move/remove image to/from split view
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        # Store the callback in an instance variable
        self.mouse_move_callback = self._get_mouse_coordinates
        self.viewer.mouse_move_callbacks.append(self.mouse_move_callback)

    def reset_view(self):
        self.viewer.reset_view()
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _sync_view(self):
        # synce cammera zoom from base viewer
        # self.viewer_model2.camera.zoom = self.viewer.camera.zoom
        try:
            layer_index = self.viewer_model1.layers.index(self.points_layer)
            self.viewer_model1.layers.move(layer_index, -1)
        except:
            pass

    def _get_mouse_coordinates(self, viewer, event):
        # Get mouse position
        points = np.round(event.position).astype(np.int32)
        points = np.where(points < 0, 0, points)

        # Update the points layer in the target viewer with the mapped position
        if len(points) == 2:
            points = (0, points[0], points[1])
        self.points_layer.data = [points]

        # Update zoom
        self.viewer_model1.camera.zoom = 10
        self.viewer_model1.camera.center = points
        self.viewer_model1.dims.set_point(0, points[0])

        # self.viewer_model2.camera.zoom = 2
        # self.viewer_model2.camera.center = points
        self.viewer_model2.dims.set_point(0, points[0])

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            self.viewer_model1.layers.selection.active = None
            self.viewer_model2.layers.selection.active = None
            return

        if event.value.name in self.viewer_model1.layers:
            self.viewer_model1.layers.selection.active = self.viewer_model1.layers[
                event.value.name
            ]

        if event.value.name in self.viewer_model2.layers:
            self.viewer_model2.layers.selection.active = self.viewer_model2.layers[
                event.value.name
            ]

    def _order_update(self):
        order = list(self.viewer.dims.order)
        # if len(order) <= 2:
        self.viewer_model1.dims.order = order
        self.viewer_model2.dims.order = order
        return

        # order[-3:] = order[-2], order[-3], order[-1]
        # self.viewer_model1.dims.order = order
        # order = list(self.viewer.dims.order)
        # order[-3:] = order[-1], order[-2], order[-3]
        # self.viewer_model2.dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""
        try:
            self.viewer_model1.layers.insert(
                event.index, copy_layer(event.value, "View_1")
            )
        except:
            pass
        try:
            self.viewer_model2.layers.insert(
                event.index, copy_layer_viewer2(event.value, "View_2")
            )
        except:
            pass

        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                OwnPartial(self._property_sync, name)
            )

        if isinstance(event.value, Points):
            event.value.events.set_data.connect(self._set_data_refresh)
            if event.value.name in self.viewer_model1.layers:
                self.viewer_model1.layers[event.value.name].events.set_data.connect(
                    self._set_data_refresh
                )
            # if event.value.name ==
            # self.viewer_model2.layers[event.value.name].events.set_data.connect(
            #     self._set_data_refresh
            # )
        event.value.events.name.connect(self._sync_name)

        self._order_update()

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = (
            event.new_index if event.new_index < event.index else event.new_index + 1
        )
        self.viewer_model1.layers.move(event.index, dest_index)
        self.viewer_model2.layers.move(event.index, dest_index)

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        if event.value.name in self.viewer_model1.layers:
            self.viewer_model1.layers.pop(event.index)
        if event.value.name in self.viewer_model2.layers:
            self.viewer_model2.layers.pop(event.index)

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if event.source.name in model.layers:
                layer = model.layers[event.source.name]
                if layer is event.source:
                    continue
                try:
                    self._block = True
                    layer.refresh()
                finally:
                    self._block = False

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        self.viewer_model1.layers[index].name = event.source.name
        self.viewer_model2.layers[index].name = event.source.name

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            if event.source.name in self.viewer_model1.layers:
                setattr(
                    self.viewer_model1.layers[event.source.name],
                    name,
                    getattr(event.source, name),
                )
            # check if layer exists in viewer2
            if event.source.name in self.viewer_model2.layers:
                setattr(
                    self.viewer_model2.layers[event.source.name],
                    name,
                    getattr(event.source, name),
                )
        finally:
            self._block = False


class AnnotationWidgetv2(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(layout="vertical")
        self.napari_viewer = napari_viewer

        # Global
        self.color_map_specified = {0.0: "#D81B60", 1.0: "#1E88E5", 2.0: "#FFC107"}
        self.activate_click = False
        self.image_layer_name = ""
        self.particle = []
        self.selected_particle_id = None
        self.filename = None
        self.cur_proposal_index, self.proposals = 0, []
        self.model, self.model_pred, self.weights, self.bias = None, None, None, None
        self.init = False
        self.AL_weights = None
        self.chosen_particles = []
        self.curr_layer = "Chosen Particles"
        self.true_labels = np.array([])

        # Key binding
        try:
            self.napari_viewer.bind_key("z", self.ZEvent)
            self.napari_viewer.bind_key("x", self.XEvent)
            self.napari_viewer.bind_key("c", self.CEvent)
            self.napari_viewer.bind_key("s", self.SEvent)
            self.napari_viewer.bind_key("d", self.DEvent)
        except ValueError:
            pass

        if self.track_mouse_position not in self.napari_viewer.mouse_move_callbacks:
            self.napari_viewer.mouse_move_callbacks.append(self.track_mouse_position)

        if (
            self.move_selected_point
            not in self.napari_viewer.mouse_double_click_callbacks
        ):
            self.napari_viewer.mouse_double_click_callbacks.append(
                self.move_selected_point
            )
        self.mouse_position = None
        self.click_add_point_callback = None

        # Initialize model
        self.load_ALM = PushButton(name="Load model")
        self.load_ALM.clicked.connect(self._load_model)
        self.save_ALM = PushButton(name="Save model")
        self.save_ALM.clicked.connect(self._save_model)

        # spacer1 = Label(value="------- Step 1: Initialize New Dataset ------")
        self.sampling_layer = LineEdit(name="Pixel", value="8.0")
        self.box_size = LineEdit(name="Box", value="5")
        self.patch_size = LineEdit(name="Patch", value="128")

        self.temp_id = LineEdit(name="pdb ID", value="7A4M")

        self.init_data = PushButton(name="Initialize dataset")
        self.init_data.clicked.connect(self._initialize_dataset)

        self.choose_init = PushButton(name="Choose Initial picks")
        self.choose_init.clicked.connect(self._choose_initial_picks)

        self.save = PushButton(name="Save Model")
        self.save.clicked.connect(self._save_model)
        self.load = PushButton(name="Load Model")
        self.load.clicked.connect(self._load_model)

        self.export = PushButton(name="Export Coordinates")
        self.export.clicked.connect(self._export_coordinates)

        self.import_ = PushButton(name="Import Coordinates")
        self.import_.clicked.connect(self._import_coordinates)

        # spacer2 = Label(value="------ Step 2: Initialize Active learning model -------")
        self.patch = PushButton(name="Change Patch")
        self.patch.clicked.connect(self._change_patch)
        self.refresh = PushButton(name="Retrain")
        self.refresh.clicked.connect(self._refresh)
        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)

        # spacer3 = Label(value="------------ Step 3: Visualize labels tool ------------")
        self.slide_pred = FloatSlider(
            name="Filter Particle",
            min=0,
            max=1,
        )
        self.slide_pred.changed.connect(self.filter_particle)

        self.points_layer = create_widget(annotation=Points, label="ROI", options={})
        self.points_layer.changed.connect(self._update_roi_info)

        self.component_selector = SpinBox(name="Particle ID", min=0)
        self.component_selector.changed.connect(self._component_num_changed)

        # self.zoom_factor = create_widget(
        #     annotation=float, label="Zoom factor", value=100
        # )
        # self.zoom_factor.changed.connect(self._component_num_changed)
        #
        # self.reset_view = PushButton(name="Reset View")
        # self.reset_view.clicked.connect(self._reset_view)

        # spacer4 = Label(value="------------ Step 4: Manual labels tool ------------")
        self.manual_label = PushButton(name="Gaussian pre-process")
        self.manual_label.clicked.connect(self.initialize_labeling)
        layer_init = VBox(
            widgets=(
                HBox(
                    widgets=(
                        self.sampling_layer,
                        self.box_size,
                    )
                ),
                HBox(
                    widgets=(
                        self.patch_size,
                        self.temp_id,
                    )
                ),
                HBox(widgets=(self.choose_init, self.init_data)),
                # HBox(widgets=(self.recenter_positive, self.show_tm_scores)),
                HBox(
                    widgets=(
                        self.save,
                        self.load,
                    )
                ),
            )
        )
        layer_AL = VBox(
            widgets=(HBox(widgets=(self.patch, self.refresh)), self.predict)
        )
        layer_slider = HBox(widgets=(self.slide_pred,))
        layer_visual1 = HBox(widgets=(self.points_layer, self.component_selector))
        # layer_visual2 = HBox(widgets=(self.zoom_factor, self.reset_view))

        label = VBox(
            widgets=(
                HBox(widgets=(self.export, self.import_)),
                HBox(widgets=(self.manual_label,)),
            )
        )
        # self.insert(0, layout_model)
        # self.insert(1, spacer1)
        self.insert(1, layer_init)
        # self.insert(3, spacer2)
        self.insert(2, layer_AL)
        # self.insert(5, spacer3)
        self.insert(3, layer_slider)
        self.insert(4, layer_visual1)
        # self.insert(5, layer_visual2)
        # self.insert(9, spacer4)
        self.insert(5, label)

        device_ = get_device()
        show_info(f"Active learning model runs on: {device_}")

    def _load_model(self):
        """Logic to load pre-train active learning model"""
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        self.AL_weights = torch.load(f"{self.filename}")

    def _save_model(self):
        """Logic to save pre-train active learning model"""
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File", directory="Active_learn_model.pth"
        )
        if self.model is not None:
            torch.save([self.model.weights, self.model.bias], filename)

    def _import_coordinates(self):
        """Logic to import coordinates"""
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        try:
            data, labels = load_coordinates(self.filename)
            self.true_labels = update_true_labels(self.true_labels, data[:, 1:], labels)
            self.create_point_layer(data[:, 1:], labels, name="Imported Labels")
            print("Loaded coordinates")
        except:
            show_info("Could not load coordinates!")

    def _export_coordinates(self):
        """Logic to export coordinates"""
        # try:
        points_layer = self.napari_viewer.layers["Initial_Labels"].data
        label = self.napari_viewer.layers["Initial_Labels"].properties["label"]
        print(points_layer.shape, label.shape)
        print(f"{self.image_layer_name}_Prediction")

        pred_points = self.napari_viewer.layers[
            f"{self.image_layer_name}_Prediction"
        ].data
        pred_label = self.napari_viewer.layers[
            f"{self.image_layer_name}_Prediction"
        ].properties["label"]
        print(pred_points.shape, pred_label.shape)

        points_layer = np.vstack((points_layer, pred_points))
        label = np.hstack((label, pred_label))

        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File", directory="coordinates.csv"
        )
        if self.true_labels.size:
            # dont save first axis
            save_coordinates(filename, np.hstack((label[:, None], points_layer)))
            print("Saved coordinates")

    def _choose_initial_picks(self):
        self.image_layer_name = self.napari_viewer.layers.selection.active.name
        self.curr_layer = "Chosen Particles"
        self.create_point_layer(np.array([]), np.array([]), name="Chosen Particles")
        self.activate_click = True

    def _initialize_dataset(self):
        self.init = True

        try:
            self.chosen_particles = self.napari_viewer.layers["Chosen Particles"].data
            # labels = self.napari_viewer.layers["Chosen Particles"].properties["label"]

            # choose only particles with label 1
            # self.chosen_particles = self.chosen_particles[labels == 1]
            assert len(self.chosen_particles) > 0
            self.napari_viewer.layers.remove("Chosen Particles")
        except:
            show_info(f"Please choose few particles to initialize the model!")
            return
        self.activate_click = True
        self.curr_layer = "Initial_Labels"

        # Image data
        try:
            assert self.image_layer_name != ""
            active_layer_name = self.image_layer_name
            # active_layer_name = self.napari_viewer.layers.selection.active.name
            # self.image_layer_name = active_layer_name
        except:
            show_info("Please load and select image!")
            return
        img = self.napari_viewer.layers[active_layer_name]

        """Down_sample dataset"""
        self.img_process = img.data
        factor = float(self.sampling_layer.value) / 8  # Normalize to 8A
        self.img_process = downsample(img.data, factor=factor)

        self.shape = self.img_process.shape
        self.img_process, _ = normalize(
            self.img_process.copy(), method="affine", use_cuda=False
        )
        img.data = self.img_process

        self.napari_viewer.layers[active_layer_name].contrast_limits = (
            self.img_process.min(),
            self.img_process.max(),
        )

        # Initialize dataset
        if self.img_process.ndim == 3:
            # Initialize template here
            tm_scores = load_template()
            tm_scores = downsample(tm_scores, factor=factor)
            # for idx, i in enumerate(tm_scores):
            #     min_ = i.min()
            #     max_ = i.max()
            #     tm_scores[idx, :] = (i - min_) / (max_ - min_)

            # tm_scores, _ = normalize(tm_scores.copy(), method="affine", use_cuda=False)
            self.tm_scores = tm_scores

            # show only chosen particle's TM scores
            self.create_image_layer(self.tm_scores[0])

            # self.tm_scores = np.zeros(self.img_process.shape)
            self.patch_corner = get_random_patch(
                self.img_process.shape, int(self.patch_size.value), self.chosen_particles
            )

            patch = self.img_process[
                self.patch_corner[0] : self.patch_corner[0]
                + int(self.patch_size.value),
                self.patch_corner[1] : self.patch_corner[1]
                + int(self.patch_size.value),
                self.patch_corner[2] : self.patch_corner[2]
                + int(self.patch_size.value),
            ]
            tm_score = self.tm_scores[
                :,
                self.patch_corner[0] : self.patch_corner[0]
                + int(self.patch_size.value),
                self.patch_corner[1] : self.patch_corner[1]
                + int(self.patch_size.value),
                self.patch_corner[2] : self.patch_corner[2]
                + int(self.patch_size.value),
            ]
            self.shape = patch.shape
            self.x = torch.from_numpy(tm_score.copy()).float().permute(1, 2, 3, 0)
            self.x = self.x.reshape(-1, self.x.shape[-1])
        else:
            self.x, _, _ = initialize_model(self.img_process)
            self.patch_corner = None
            self.x = torch.from_numpy(self.x)

        self.model = BinaryLogisticRegression(
            n_features=self.x.shape[1], l2=1.0, pi=0.01, pi_weight=1000
        )

        """Initialize new model or load pre-trained"""
        # update y and count
        self.y = label_points_to_mask([], self.shape, self.box_size.value)
        self.count = torch.where(
            ~torch.isnan(self.y), torch.ones_like(self.y), torch.zeros_like(self.y)
        )
        self.model.fit(
            self.x,
            self.y.ravel(),
            weights=self.count.ravel(),
            pre_train=self.AL_weights,
        )

        self.proposals = []
        self.proposals, scores = find_peaks(
            tm_score[0, :], int(self.box_size.value), with_score=True
        )
        order = np.argsort(scores)
        self.proposals = self.proposals[order]

        # Add point which model are least certain about
        # idx = np.random.choice(len(self.proposals), 10, replace=False)
        # points = np.array([self.proposals[i] for i in idx])
        points = np.vstack(self.proposals[:10])
        points = correct_coord(points, self.patch_corner, True)
        labels = np.zeros((points.shape[0],))
        labels[:] = 2

        self.create_point_layer(points.astype(np.float64), labels)
        """ Initialize model and pick initial particles """
        self.activate_click = True
        try:
            self.napari_viewer.selection.active = self.napari_viewer.layers[
                "Initial_Labels"
            ]
        except:
            pass

        self.reset_view()
        show_info("Task finished: Initialize Dataset!")

    def _change_patch(self):
        self.curr_layer = "Initial_Labels"
        if self.patch_corner is not None:
            # Re-Train model
            self.activate_click = True

            points_layer = self.napari_viewer.layers["Initial_Labels"].data
            label = self.napari_viewer.layers["Initial_Labels"].properties["label"]
            if np.any(label == 2):
                show_info(f"Please Correct all uncertain particles!")
                return

            self.true_labels = update_true_labels(self.true_labels, points_layer, label)
            points_layer = correct_coord(points_layer, self.patch_corner, False)

            # # correct coord of all existing true_labels
            # if self.true_labels.size:
            #     x = self.true_labels[:, 1:].copy()
            #     prev_labels = correct_coord(x, self.patch_corner, False)
            #     prev_labels = np.array(
            #         (
            #             self.true_labels[:, 0],
            #             prev_labels[:, 0],
            #             prev_labels[:, 1],
            #             prev_labels[:, 2],
            #         )
            #     ).T

            data = np.asarray(points_layer)
            if data.shape[1] == 2:
                data = np.array(
                    (np.array(label).astype(np.int16), data[:, 0], data[:, 1])
                ).T
            else:
                data = np.array(
                    (
                        np.array(label).astype(np.int16),
                        data[:, 0],
                        data[:, 1],
                        data[:, 2],
                    )
                ).T

            # add true_labels from the global variable to the data; only consider positive x,y,z
            # try:
            #     if prev_labels.size:
            #         for data_ in prev_labels:
            #             # if all values in data_ are positive
            #             # check if already not there
            #             if np.any(np.all(data_ == data, axis=1)) == False:
            #                 if np.all(data_ > 0):
            #                     data = np.vstack((data, data_))
            #                     self.count += 1
            # except:
            #     pass

            self.y = label_points_to_mask(data, self.shape, self.box_size.value)
            self.count = (~torch.isnan(self.y)).float()

            self.model.fit(self.x, self.y.ravel(), weights=self.count.ravel())

            show_info("Task finished: Retrain model!")

            # Feed new patch
            self.patch_corner = get_random_patch(
                self.img_process.shape, int(self.patch_size.value), self.chosen_particles
            )
            print(self.patch_corner)
            patch = self.img_process[
                self.patch_corner[0] : self.patch_corner[0]
                + int(self.patch_size.value),
                self.patch_corner[1] : self.patch_corner[1]
                + int(self.patch_size.value),
                self.patch_corner[2] : self.patch_corner[2]
                + int(self.patch_size.value),
            ]
            tm_score = self.tm_scores[
                :,
                self.patch_corner[0] : self.patch_corner[0]
                + int(self.patch_size.value),
                self.patch_corner[1] : self.patch_corner[1]
                + int(self.patch_size.value),
                self.patch_corner[2] : self.patch_corner[2]
                + int(self.patch_size.value),
            ]
            self.shape = patch.shape

            # Features from new patch
            self.x = torch.from_numpy(tm_score.copy()).float().permute(1, 2, 3, 0)
            self.x = self.x.reshape(-1, self.x.shape[-1])

            # ToDo Use here rank_candidate_locations() set proposals to [] and
            #  use this method to get entropy scores to pick top 10 entropy
            with torch.no_grad():
                logits = self.model(self.x).reshape(*self.shape)
            logits = logits.cpu().detach()
            print(logits.shape)
            self.create_image_layer(logits.cpu().detach().numpy(), "logits")

            """Get Entropy"""
            # rank_candidate_locations()
            self.proposals = []
            self.proposals = rank_candidate_locations(logits, self.shape)

            # Add point which model are least certain about
            points = np.vstack(self.proposals[:10])
            points = correct_coord(points, self.patch_corner, True)

            labels = np.zeros((points.shape[0],))
            labels[:] = 2

            pos_points = np.vstack(self.proposals[:10])
            pos_labels = np.zeros((pos_points.shape[0],))
            pos_labels[:] = 1

            self.create_point_layer(points, labels)
            self.activate_click = True

            try:
                self.napari_viewer.selection.active = self.napari_viewer.layers[
                    "Initial_Labels"
                ]
            except:
                pass

            # attempt to remove the old layer
            # try:
            #     self.napari_viewer.layers.remove("Predicted_Labels")
            # except:
            #     pass

            # self.napari_viewer.add_points(
            #     pos_points,
            #     name="Predicted_Labels",
            #     properties={"confidence": pos_labels},
            #     edge_color="black",
            #     face_color="confidence",
            #     face_colormap="viridis",
            #     edge_width=0.1,
            #     symbol="disc",
            #     size=5,
            # )

            self.reset_view()

            self.component_selector.value = 0

            show_info(f"Task finished: Drawn new patch!")

    def _refresh(self):
        """
        Re-train model, and add 10 most uncertain points to the list
        """
        self.activate_click = True
        self.curr_layer = "Initial_Labels"
        points_layer = self.napari_viewer.layers["Initial_Labels"].data
        label = self.napari_viewer.layers["Initial_Labels"].properties["label"]
        self.true_labels = update_true_labels(self.true_labels, points_layer, label)

        points_layer = correct_coord(points_layer, self.patch_corner, False)

        # correct coord of all existing true_labels
        if self.true_labels.size:
            x = self.true_labels[:, 1:].copy()
            prev_labels = correct_coord(x, self.patch_corner, False)
            prev_labels = np.array(
                (
                    self.true_labels[:, 0],
                    prev_labels[:, 0],
                    prev_labels[:, 1],
                    prev_labels[:, 2],
                )
            ).T

        if self.patch_corner is not None:
            points_layer = correct_coord(points_layer, self.patch_corner, False)
            if self.true_labels.size:
                prev_labels = correct_coord(self.true_labels, self.patch_corner, False)

        if np.any(label == 2):
            show_info(f"Please Correct all uncertain particles!")
        else:
            data = np.asarray(points_layer)
            if data.shape[1] == 2:
                data = np.array(
                    (np.array(label).astype(np.int16), data[:, 0], data[:, 1])
                ).T
            else:
                data = np.array(
                    (
                        np.array(label).astype(np.int16),
                        data[:, 0],
                        data[:, 1],
                        data[:, 2],
                    )
                ).T

            # add true_labels from the global variable to the data; remove if any is negative
            try:
                if prev_labels.size:
                    for data_ in prev_labels:
                        # if all values in data_ are positive
                        # check if already not there
                        if np.any(np.all(data_ == data, axis=1)) == False:
                            if np.all(data_ > 0):
                                data = np.vstack((data, data_))
                                self.count += 1
            except:
                pass

            self.y = label_points_to_mask(data, self.shape, self.box_size.value)

            self.count = (~torch.isnan(self.y)).float()
            self.model.fit(self.x, self.y.ravel(), weights=self.count.ravel())

            self.cur_proposal_index, self.proposals = rank_candidate_locations(
                self.model, self.x, self.shape, self.proposals, id_=1
            )

            # Add point which model are least certain about
            points = np.vstack(self.proposals[:10])
            label_unknown = np.zeros((points.shape[0],))
            label_unknown[:] = 2

            pos_points = np.vstack(self.proposals[:10])
            pos_labels = np.zeros((pos_points.shape[0],))
            pos_labels[:] = 1

            # attempt to remove the old layer
            try:
                self.napari_viewer.layers.remove("Predicted_Labels")
            except:
                pass

            data = np.vstack((data[:, 1:], points.astype(np.float64)))
            if self.patch_corner is not None:
                data = correct_coord(data, self.patch_corner, True)

            labels = np.hstack((label, label_unknown))
            self.create_point_layer(data, labels)

            self.napari_viewer.add_points(
                pos_points,
                name="Predicted_Labels",
                properties={"confidence": pos_labels},
                edge_color="black",
                face_color="confidence",
                face_colormap="viridis",
                edge_width=0.1,
                symbol="disc",
                size=5,
            )

            show_info("Task finished: Retrain model!")

    def _predict(self):
        self.curr_layer = "Initial_Labels"
        self.activate_click = False

        if not self.init:
            self.init = True

            active_layer_name = self.napari_viewer.layers.selection.active.name
            self.image_layer_name = active_layer_name
            img = self.napari_viewer.layers[active_layer_name]

            """Down_sample dataset"""
            factor = float(self.sampling_layer.value) / 8
            self.img_process = downsample(img.data, factor=factor)

            self.shape = self.img_process.shape
            self.img_process, _ = normalize(
                self.img_process, method="gmm", use_cuda=False
            )
            img.data = self.img_process

            self.napari_viewer.layers[active_layer_name].contrast_limits = (
                self.img_process.min(),
                self.img_process.max(),
            )

            # Load TM scores if not loaded already
            tm_scores = load_template(self.image_layer_name, self.temp_id.value)
            # downsample tm_scores
            factor = float(factor) / 8
            tm_scores = downsample(tm_scores.squeeze(0), factor=factor)
            tm_scores, _ = normalize(tm_scores.copy(), method="affine", use_cuda=False)
            self.tm_scores = tm_scores

            # Initialize dataset
            if self.img_process.ndim == 2:
                self.x, _, p_label = initialize_model(
                    self.img_process, img_name=self.image_layer_name
                )

                self.y = label_points_to_mask([], self.shape, self.box_size.value)
                self.count = torch.where(
                    ~torch.isnan(self.y),
                    torch.ones_like(self.y),
                    torch.zeros_like(self.y),
                )

                self.model = BinaryLogisticRegression(
                    n_features=self.x.shape[1], l2=1.0, pi=0.01, pi_weight=1000
                )

                if self.AL_weights is not None:
                    self.model.fit(
                        self.x,
                        self.y.ravel(),
                        weights=self.count.ravel(),
                        pre_train=self.AL_weights,
                    )

        # if any label is 2
        if np.any(self.napari_viewer.layers["Initial_Labels"].properties["label"] == 2):
            show_info(f"Please Correct all uncertain particles!")
            self.activate_click = True
            return

        if self.img_process.ndim == 2:
            with torch.no_grad():
                logits = self.model(self.x).reshape(*self.shape)
            logits = logits.numpy()

            max_filter = maximum_filter(logits, size=self.box_size.value)
            peaks = logits - max_filter
            peaks = np.where(peaks == 0)
            peaks = np.stack(peaks, axis=-1)
            if peaks.shape[1] == 3:
                peak_logits = logits[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
            else:
                peak_logits = logits[peaks[:, 0], peaks[:, 1]]
        else:
            peaks, peak_logits = predict_3d_with_AL(
                self.img_process,
                self.model,
                self.AL_weights,
                int(self.patch_size.value),
                tm_scores=self.tm_scores,
            )

            # [UNCOMMENT THIS AFTER TESTING]

            # add self.true_labels[:,1:] to the peaks
            # max_logits = np.max(peak_logits)
            # if self.true_labels.size:
            #     peaks = np.vstack((peaks, self.true_labels[:, 1:]))
            #     peak_logits = np.hstack(
            #         (peak_logits, max_logits * np.ones(self.true_labels.shape[0]))
            #     )

        print(peaks.shape, peak_logits.shape)
        self.napari_viewer.add_points(
            peaks,
            name=f"{self.image_layer_name}_Prediction",
            properties={"confidence": peak_logits},
            edge_color="black",
            face_color="confidence",
            face_colormap="viridis",
            edge_width=0.1,
            symbol="disc",
            size=5,
        )
        # if self.show_tm_scores.value:
        #     self.create_image_layer(self.tm_scores)
        #     # set prediction layer as active layer
        #     self.napari_viewer.selection.active = self.napari_viewer.layers[
        #         f"{self.image_layer_name}_Prediction"
        #     ]
        self.reset_view()

        try:
            self.napari_viewer.layers["Initial_Labels"].visible = False
        except:
            pass
        self.slide_pred.value = 0

        self.particle = peaks
        self.confidence = peak_logits
        self.slide_pred.min = np.min(self.confidence)
        self.slide_pred.max = np.max(self.confidence)

        show_info(f"Task finished: Particle peaking!")

    def filter_particle(self):
        if len(self.particle) > 0:
            active_layer_name = self.napari_viewer.layers.selection.active.name
            if active_layer_name.endswith("Prediction_Filtered"):
                self.napari_viewer.layers.remove(active_layer_name)
                active_layer_name = active_layer_name[:-20]
            self.napari_viewer.layers[f"{active_layer_name}"].visible = False

            keep_id = np.where(self.confidence >= self.slide_pred.value)

            filter_particle = self.particle[keep_id[0], :]
            filter_confidence = self.confidence[keep_id[0]]

            self.napari_viewer.add_points(
                filter_particle,
                name=f"{active_layer_name}_Prediction_Filtered",
                properties={"confidence": filter_confidence},
                edge_color="black",
                face_color="confidence",
                face_colormap="viridis",
                edge_width=0.1,
                symbol="disc",
                size=5,
            )

    def initialize_labeling(self, viewer):
        self.activate_click = True

        # Image data
        active_layer_name = self.napari_viewer.layers.selection.active.name
        self.image_layer_name = active_layer_name
        img = self.napari_viewer.layers[active_layer_name]

        """Down_sample dataset"""
        factor = float(self.sampling_layer.value) / 8
        self.img_process = downsample(img.data, factor=factor)

        self.shape = self.img_process.shape
        img.data, _ = normalize(self.img_process, method="affine", use_cuda=False)

        self.napari_viewer.layers[active_layer_name].contrast_limits = (
            img.data.min(),
            img.data.max(),
        )

        img.data = nd.gaussian_filter(img.data, 2)

    def ZEvent(self, viewer):
        name = self.curr_layer
        if self.activate_click:
            points_layer = viewer.layers[name].data
            if points_layer.shape[0] == 0:
                self.update_point_layer_2(self.mouse_position, 0, "add")
            else:
                # Calculate the distance between the mouse position and all points
                kdtree = KDTree(points_layer)
                distance, closest_point_index = kdtree.query(self.mouse_position, k=1)

                if distance > 10:
                    self.update_point_layer_2(self.mouse_position, 0, "add")
                else:
                    self.update_point_layer_2(closest_point_index, 0, "update")

    def XEvent(self, viewer):
        name = self.curr_layer
        if self.activate_click:
            points_layer = viewer.layers[name].data
            if points_layer.shape[0] == 0:
                self.update_point_layer_2(self.mouse_position, 1, "add")
            else:
                # Calculate the distance between the mouse position and all points
                kdtree = KDTree(points_layer)
                distance, closest_point_index = kdtree.query(self.mouse_position, k=1)

                if distance > 12:
                    self.update_point_layer_2(self.mouse_position, 1, "add")
                else:
                    self.update_point_layer_2(closest_point_index, 1, "update")

    def CEvent(self, viewer):
        name = self.curr_layer
        if self.activate_click:
            points_layer = viewer.layers[name].data

            # Calculate the distance between the mouse position and all points
            kdtree = KDTree(points_layer)
            distance, closest_point_index = kdtree.query(self.mouse_position, k=1)

            self.update_point_layer_2(closest_point_index, 0, "remove")

    def SEvent(self, viewer):
        if self.activate_click:
            if self.component_selector.value > 0:
                # self.reset_view()
                self.component_selector.value = self.component_selector.value - 1

    def DEvent(self, viewer):
        name = self.curr_layer
        try:
            points_layer = len(viewer.layers[name].data)

            if self.component_selector.value < points_layer:
                # self.reset_view()
                self.component_selector.value = self.component_selector.value + 1
        except KeyError:
            pass

    def track_mouse_position(self, viewer, event):
        self.mouse_position = event.position

    def move_selected_point(self, viewer, event):
        if self.activate_click:
            try:
                # if self.activate_click:
                points_layer = viewer.layers["Initial_Labels"].data

                # Calculate the distance between the mouse position and all points
                distances = np.linalg.norm(points_layer - self.mouse_position, axis=1)
                closest_point_index = distances.argmin()

                # Clear the current selection and Select the closest point
                if self.selected_particle_id != closest_point_index:
                    self.selected_particle_id = closest_point_index

                    viewer.layers["Initial_Labels"].selected_data = set()
                    viewer.layers["Initial_Labels"].selected_data.add(
                        closest_point_index
                    )
            except:
                pass

    def create_point_layer(self, point, label, name="Initial_Labels"):
        try:
            self.napari_viewer.layers.remove(name)
        except:
            pass
        if point.shape[0] > 0:
            self.napari_viewer.add_points(
                point,
                name=name,
                face_color="#00000000",
                properties={"label": label.astype(np.int16)},
                edge_color="label",
                edge_color_cycle=self.color_map_specified,
                edge_width=0.1,
                symbol="square",
                size=40,
            )
            self.napari_viewer.layers[name].mode = "select"
        else:
            self.napari_viewer.add_points(
                point,
                name=name,
                face_color="#00000000",
                edge_color_cycle=self.color_map_specified,
                edge_width=0.1,
                symbol="square",
                size=40,
            )
            self.napari_viewer.layers[name].mode = "select"

    def create_image_layer(self, tm_scores, name="TM_Scores"):
        try:
            self.napari_viewer.layers.remove(name)
        except:
            pass

        print("Creating image layer for tm_scores")
        self.napari_viewer.add_image(
            tm_scores, name=name, colormap="viridis", opacity=0.25
        )

        try:
            self.napari_viewer.layers[name].contrast_limits = (
                tm_scores.min(),
                tm_scores.max(),
            )
        except:
            pass

        # set layer as not visible
        self.napari_viewer.layers[name].visible = False

    def update_point_layer_2(self, index, label, func):
        name = self.curr_layer
        p_layer = self.napari_viewer.layers[name]

        if func == "add":
            point_layer = p_layer.data
            if point_layer.shape[0] == 0:
                labels = np.array([label])
                points_layer = np.array([self.mouse_position])
            else:
                labels = p_layer.properties["label"]
                points_layer = np.insert(point_layer, 0, self.mouse_position, axis=0)
                labels = np.insert(labels, 0, [label], axis=0)

            self.create_point_layer(points_layer, labels, name)

        elif func == "remove":
            point_layer = p_layer.data
            labels = p_layer.properties["label"]

            points_layer = np.delete(point_layer, index, axis=0)
            labels = np.delete(labels, index, axis=0)

            self.create_point_layer(points_layer, labels, name)
        elif func == "update":
            point_layer = p_layer.data
            labels = p_layer.properties["label"]
            if labels[index] != label:
                labels[index] = label
                self.create_point_layer(point_layer, labels, name)
        # else:
        #     pass
        p_layer.edge_color_cycle = self.color_map_specified
        # except:
        #     pass

    def _update_roi_info(self):
        if not self.activate_click:
            self._component_num_changed()

    def _component_num_changed(self):
        self._zoom()

    def _zoom(self):
        if self.napari_viewer.dims.ndisplay != 2:
            show_info("Zoom in does not work in 3D mode")

        num = self.component_selector.value
        if num >= len(self.points_layer.value.data):
            num = len(self.points_layer.value.data) - 1

        points = self.points_layer.value.data
        if len(points) > 0:
            points = np.round(self.points_layer.value.data[num]).astype(np.int32)
            points = np.where(points < 0, 0, points)

            lower_bound = points - 1
            lower_bound = np.where(lower_bound < 0, 0, lower_bound)
            upper_bound = points + 1
            upper_bound = np.where(upper_bound < 0, 0, upper_bound)
            diff = upper_bound - lower_bound
            frame = diff * 99

            if self.napari_viewer.dims.ndisplay == 2:
                rect = Rect(
                    pos=(lower_bound - frame)[-2:][::-1],
                    size=(diff + 2 * frame)[-2:][::-1],
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Public access to Window.qt_viewer"
                    )
                    self.napari_viewer.window.qt_viewer.view.camera.set_state(
                        {"rect": rect}
                    )
            self._update_point(lower_bound, upper_bound)

    def _update_point(self, lower_bound, upper_bound):
        point = (lower_bound + upper_bound) / 2
        current_point = self.napari_viewer.dims.point[-len(lower_bound) :]
        dims = len(lower_bound) - self.napari_viewer.dims.ndisplay
        start_dims = self.napari_viewer.dims.ndim - len(lower_bound)

        for i in range(dims):
            if not (lower_bound[i] <= current_point[i] <= upper_bound[i]):
                self.napari_viewer.dims.set_point(start_dims + i, point[i])

    def _show_tm_scores(self):
        if self.show_tm_scores.value:
            try:
                self.napari_viewer.layers["TM_Scores"].visible = True
            except:
                pass
        else:
            try:
                self.napari_viewer.layers["TM_Scores"].visible = False
            except:
                pass

    def reset_view(self):
        self.napari_viewer.reset_view()
