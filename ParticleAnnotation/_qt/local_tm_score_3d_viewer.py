import numpy as np
import pandas as pd
import starfile

from napari import Viewer
from napari.utils.notifications import show_info
from napari_bbox import BoundingBoxLayer

from sklearn.neighbors import KDTree

import torch
from topaz.stats import normalize

from magicgui.widgets import (
    Container,
    PushButton,
    LineEdit,
    ComboBox,
    FloatSlider,
    Label,
    VBox,
    HBox,
)
from qtpy.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QDialog, QVBoxLayout

from ParticleAnnotation.utils.load_data import (
    load_template,
    load_coordinates,
    load_tomogram,
)
from ParticleAnnotation.utils.model.active_learning_model import (
    BinaryLogisticRegression,
    label_points_to_mask,
    predict_3d_with_AL,
    stack_all_labels,
)
from ParticleAnnotation.utils.model.utils import (
    correct_coord,
    find_peaks,
    get_device,
    get_random_patch,
    rank_candidate_locations,
)
from ParticleAnnotation.utils.viewer.viewer_functionality import (
    build_gird_with_particles,
    draw_patch_and_scores,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

colormap_for_display = "Spectral"


class AnnotationWidget(Container):
    def __init__(self, viewer_tm_score_3d: Viewer):
        super().__init__(layout="vertical")
        self.napari_viewer = viewer_tm_score_3d
        self.delta_plot = PlotPopup()
        self.delta_plot.show()

        # Global
        self.image_name = ""
        self.filename = None
        self.tm_list = None
        self.img_process, self.patch_corner = None, None

        # Particles selections
        self.cur_proposal_index, self.proposals = 0, []
        self.user_annotations = np.zeros((0, 4))  # Z, Y, X, Label
        self.selected_particle_id = None

        # Remove after testing
        self.particle = None
        self.confidence = None
        self.patch_points, self.patch_label = np.zeros((0, 3)), np.zeros((1,))

        # BLR model
        self.model, self.model_pred, self.weights, self.bias = None, None, None, None
        self.init, self.init_done, self.AL, self.Predict = False, False, False, False
        self.AL_weights = None
        self.delta = None
        self.delta_values = []

        # Viewer
        self.color_map_particle_classes = {
            0.0: "#D81B60",  # Negative
            1.0: "#1E88E5",  # Positive
            2.0: "#FFC107",  # Unknown
        }
        self.activate_user_clicks = False
        self.correct_positions, self.patch_corner = False, None

        self.all_grid = False
        self.grid_labeling_mode = False

        # Key binding
        try:
            self.napari_viewer.bind_key(
                "z", self.ZEvent
            )  # Add/Update to Negative label
            self.napari_viewer.bind_key(
                "x", self.XEvent
            )  # Add/Update to Positive label
            self.napari_viewer.bind_key("c", self.CEvent)  # Remove label
        except ValueError:
            pass

        # Track mouse position
        if self.track_mouse_position not in self.napari_viewer.mouse_move_callbacks:
            self.napari_viewer.mouse_move_callbacks.append(self.track_mouse_position)

        if (
            self.selected_point_near_mouse
            not in self.napari_viewer.mouse_double_click_callbacks
        ):
            self.napari_viewer.mouse_double_click_callbacks.append(
                self.selected_point_near_mouse
            )

        self.mouse_position = None
        self.click_add_point_callback = None

        spacer_1 = Label(value="------------------- Options --------------------")
        self.box_size = LineEdit(name="Model Mask Size [px]", value=10)
        self.filter_size = LineEdit(name="Particle size [px]", value=15)
        self.patch_size = LineEdit(name="Region size", value=128)
        self.pdb_id = ComboBox(
            name="PDB ID:",
            value="7A4M",
            choices=(
                "1FA2",
                "1PMA",
                "6N4V",
                "6QS9",
                "6R7M",
                "7A4M",
                "7QTQ",
            ),
        )
        self.pdb_id.changed.connect(self._pdb_id_update)
        self.pi = LineEdit(name="Ratio of particles [%]", value=0.01)
        self.pi.changed.connect(self._pdb_id_update)
        self.gauss = LineEdit(name="Gaussian filter size", value=1)

        # ---------------- Import & Export modules ----------------
        self.export_particles = PushButton(name="Save particles")
        self.export_particles.clicked.connect(self._export_particles)
        self.import_particles = PushButton(name="Load particles")
        self.import_particles.clicked.connect(self._import_particles)

        spacer_2 = Label(value="------------------- Training -------------------")
        self.select_particle_for_patches = PushButton(name="Load data")
        self.select_particle_for_patches.clicked.connect(
            self._select_particle_for_patches
        )
        self.train_BLR_on_patch = PushButton(name="Re-train model")
        self.train_BLR_on_patch.clicked.connect(self._train_BLR_on_patch)

        spacer_3 = Label(value="---------------- Viewing modes -----------------")
        self.show_tomogram = PushButton(name="Tomogram")
        self.show_tomogram.clicked.connect(self._show_tomogram)
        self.show_active_learning_grid = PushButton(name="Active-Learning")
        self.show_active_learning_grid.clicked.connect(self._show_active_learning_grid)
        self.show_particle_grid = PushButton(name="Particle")
        self.show_particle_grid.clicked.connect(self._show_particle_grid)

        spacer_4 = Label(value="------------------- Predict --------------------")
        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)
        self.filter_particle_by_confidence = FloatSlider(
            name="Filter Particle",
            value=0.5,
            min=0,
            max=1,
        )
        self.filter_particle_by_confidence.changed.connect(
            self._filter_particle_by_confidence
        )

        spacer_5 = Label(value="-------------------- Output --------------------")
        self.export_particles_all = PushButton(name="All particles")
        self.export_particles_all.clicked.connect(self._export_particles_all)
        self.export_particles_labeled = PushButton(name="Human Labeled")
        self.export_particles_labeled.clicked.connect(self._export_particles_labeled)
        self.export_particles_predict = PushButton(name="Predicted Particles")
        self.export_particles_predict.clicked.connect(self._export_particles_predict)
        self.export_particles_filter = PushButton(name="Predicted & Filtered")
        self.export_particles_filter.clicked.connect(self._export_particles_filter)

        self.save_model = PushButton(name="Save Model")
        self.save_model.clicked.connect(self._save_model)
        self.load_model = PushButton(name="Load Model")
        self.load_model.clicked.connect(self._load_model)

        spacer_6 = Label(value="-------------------- Restart -------------------")
        self.reset_session = PushButton(name="Restart Session")
        self.reset_session.clicked.connect(self._reset_session)

        widget = VBox(
            widgets=(
                HBox(widgets=(spacer_1,)),
                HBox(
                    widgets=(
                        self.box_size,
                        self.filter_size,
                    )
                ),
                HBox(
                    widgets=(
                        self.pdb_id,
                        self.patch_size,
                    )
                ),
                VBox(
                    widgets=(
                        self.pi,
                        self.gauss,
                        spacer_2,
                        self.select_particle_for_patches,
                        self.train_BLR_on_patch,
                        spacer_4,
                        self.predict,
                        self.filter_particle_by_confidence,
                        spacer_3,
                    )
                ),
                HBox(
                    widgets=(
                        self.show_tomogram,
                        self.show_active_learning_grid,
                        self.show_particle_grid,
                    )
                ),
                VBox(
                    widgets=(
                        spacer_5,
                        self.import_particles,
                    )
                ),
                HBox(
                    widgets=(
                        self.export_particles_all,
                        self.export_particles_labeled,
                    )
                ),
                HBox(
                    widgets=(
                        self.export_particles_predict,
                        self.export_particles_filter,
                    )
                ),
                HBox(
                    widgets=(
                        self.save_model,
                        self.load_model,
                    )
                ),
                VBox(
                    widgets=(
                        spacer_6,
                        self.reset_session,
                    )
                ),
            )
        )

        widget.max_width = 400
        self.napari_viewer.window.add_dock_widget(widget, area="right")

        self.device_ = get_device()
        show_info(f"Active learning model runs on: {self.device_}")
        self._reset_views()

    """""" """""" """""" """
    Main triggers for GUI
    """ """""" """""" """"""

    def _select_particle_for_patches(self):
        """
        Starting function - for the plugin. Function is allowing user to label
        particles of interest, and store them.
        """
        if not self.init_done:
            self.init = True
            self.all_grid = False
            self.grid_labeling_mode = False

            # If image is not loaded, ask user to load it
            if self.napari_viewer.layers.selection.active is None:
                img, _, name = load_tomogram()

                if img is None:
                    return

                self.create_image_layer(img, name=name, transparency=False)

            self.image_name = (
                self.filename
            ) = self.napari_viewer.layers.selection.active.name
            img = self.napari_viewer.layers[self.image_name]
            self.img_process = img.data
            self.img_process, _ = normalize(
                self.img_process.copy(), method="affine", use_cuda=False
            )

            # Load and pre-process tm_scores data
            # ToDo 1 add regularizer for pushing the negative weights towards ice scores
            # add fantom data which will push the scores down. Fore each label negative array
            # ToDo 2: randome feature expansion with furier for all scores
            # Option Gaussian model
            self.tm_scores, self.tm_list = load_template()

            self._pdb_id_update()
        else:
            return

    def _train_BLR_on_patch(self):
        self.init_done = True
        self.init = False
        self.AL = True

        self._reset_views()
        self.all_grid = False
        self.grid_labeling_mode = False

        patch_size = int(self.patch_size.value)
        box_size = int(self.box_size.value)

        """If Patch do not exist create one and drawn panicles"""
        if self.patch_corner is None:
            self.patch_corner = get_random_patch(
                self.img_process.shape, patch_size, self.user_annotations[:, :3]
            )

        self.clean_viewer()
        patch, tm_score = draw_patch_and_scores(
            self.img_process, self.tm_scores, self.patch_corner, patch_size
        )

        # Initialized y (empty label mask) and count
        self.x = torch.from_numpy(tm_score.copy()).float()
        self.x = self.x.permute(1, 2, 3, 0)
        self.x = self.x.reshape(-1, self.x.shape[-1])
        self.shape = tm_score.shape[1:]

        # Take all particle crops for the training, not just a patch
        if self.model is None:
            self.model = BinaryLogisticRegression(
                n_features=self.x.shape[-1],
                l2=1.0,
                pi=float(self.pi.value),
                pi_weight=1000,
            )

        """Re-trained BLR model on user corrected particles"""
        # Retrive coordinate within patch
        stored_points = self.user_annotations.copy()[:, :3] - self.patch_corner
        point_indexes = np.all(
            (stored_points >= 0) & (stored_points <= patch_size), axis=1
        )
        points = self.user_annotations[point_indexes, :3]
        labels = self.user_annotations[point_indexes, 3]

        # Update BLR inputs
        data = np.array(
            (
                np.array(labels).astype(np.int16),
                points[:, 0],
                points[:, 1],
                points[:, 2],
            )
        ).T
        data[:, 1:] = correct_coord(data[:, 1:], self.patch_corner, False)
        self.y = label_points_to_mask(data, self.shape, box_size)
        self.count = (~torch.isnan(self.y)).float()
        self.count[self.y == 0] = 0

        data = np.hstack(
            (self.user_annotations[:, 3][:, None], self.user_annotations[:, :3])
        )

        # Data for all labels
        self.all_labels = stack_all_labels(self.tm_scores, data, box_size)

        if len(self.all_labels[0][0]) > 0:
            all_scores_pos = self.tm_scores[
                :, self.all_labels[0][0], self.all_labels[1][0], self.all_labels[2][0]
            ]
        else:
            all_scores_pos = np.ones((self.tm_scores.shape[0], 0))

        if len(self.all_labels[0][1]) > 0:
            all_scores_neg = self.tm_scores[
                :, self.all_labels[0][1], self.all_labels[1][1], self.all_labels[2][1]
            ]
        else:
            all_scores_neg = np.zeros((self.tm_scores.shape[0], 0))

        all_label_pos = np.ones(all_scores_pos.shape[1])
        all_label_neg = np.zeros(all_scores_neg.shape[1])

        if all_scores_pos.shape[1] == 0 and all_scores_neg.shape[1] == 0:
            x_filter, y_filter = None, None
        else:
            x_filter = (
                torch.from_numpy(np.hstack((all_scores_pos, all_scores_neg)))
                .float()
                .permute(1, 0)
            )
            y_filter = torch.from_numpy(
                np.concatenate((all_label_pos, all_label_neg))
            ).float()

        # Re-trained BLR model
        # Fit entire tomograms
        index_ = self.tm_idx
        x_onehot = torch.zeros(
            (x_filter.size(1), x_filter.size(1)),
            dtype=x_filter.dtype,
            device=x_filter.device,
        )

        y_onehot = torch.zeros(
            x_filter.size(1),
            dtype=y_filter.dtype,
            device=y_filter.device,
        )
        y_onehot[index_] = 1

        x_filter = torch.cat((x_filter, x_onehot), dim=0)
        y_filter = torch.cat((y_filter, y_onehot), dim=0)

        self.model.fit(
            self.x,
            self.y.ravel(),
            weights=self.count.ravel(),
            all_labels=[x_filter, y_filter],
        )
        if self.delta is None:
            show_info("Training weights delta = 0.0")
            self.delta = self.model.weights
        else:
            _delta = torch.mean(self.delta - self.model.weights)
            _delta = _delta.detach().numpy()[0]
            self.delta_values.append(torch.mean(self.delta - self.model.weights))

            show_info(f"Training weights delta = {self.delta_values[-1]}")
            print(f"Training weights delta = {self.delta_values[-1]}")

            self.delta_plot.update_plot(y_values=self.delta_values)
            self.delta = self.model.weights

        """Draw new patch and find new particle for user to label"""
        # Select patch
        self.patch_corner = get_random_patch(
            self.img_process.shape, patch_size, self.user_annotations[:, :3]
        )

        # Display new patch and associated scores
        patch, tm_score = draw_patch_and_scores(
            self.img_process, self.tm_scores, self.patch_corner, patch_size
        )

        # BLR training and model update
        self.shape = patch.shape
        self.x = torch.from_numpy(tm_score.copy()).float()
        self.x = self.x.permute(1, 2, 3, 0)
        self.x = self.x.reshape(-1, self.x.shape[-1])

        with torch.no_grad():
            logits = self.model(self.x).reshape(*self.shape)
            self.logits_patch = torch.sigmoid(logits).cpu().detach().numpy()
            logits = logits.cpu().detach()

        # Draw 10 coordinates with lowest entropy
        self.proposals = rank_candidate_locations(logits, self.shape)
        self.patch_points = np.vstack(self.proposals[:10])
        self.patch_label = np.zeros((self.patch_points.shape[0],))
        self.patch_label[:] = 2

        stored_points = self.user_annotations.copy()[:, :3] - self.patch_corner
        point_indexes = np.all(
            (stored_points >= 0) & (stored_points <= patch_size), axis=1
        )

        self._show_active_learning_grid()

    def _predict(self):
        self.AL = False
        self.Predict = True

        self.all_grid = False
        self.grid_labeling_mode = False
        self.clean_viewer()
        self._reset_views()

        if self.model is None:
            return

        patch_size = int(self.patch_size.value)
        gauss_filter = float(self.gauss.value)

        # ToDo Preiction of filamets with skeletonization and down scaling
        peaks, peaks_confidence, self.logits_full = predict_3d_with_AL(
            self.img_process,
            tm_scores=self.tm_scores,
            model=self.model,
            offset=patch_size,
            maximum_filter_size=int(self.filter_size.value),
            gauss_filter=gauss_filter,
        )
        order = np.argsort(peaks_confidence)
        self.peaks_full = peaks[order]
        self.peaks_confidence_full = peaks_confidence[order]

        self.create_image_layer(self.img_process, self.filename)
        self.create_image_layer(
            self.tm_scores[self.tm_idx], "TM_Score", transparency=True, visibility=False
        )
        self.create_image_layer(
            self.logits_full, "Sigmoid", transparency=True, range_=(0, 1)
        )

        self.create_point_layer(
            self.user_annotations[:, :3],
            self.user_annotations[:, 3],
            name="All_labels",
            visible=False,
        )

        self.napari_viewer.add_points(
            self.peaks_full,
            name="Particle_Prediction",
            properties={"label": self.peaks_confidence_full},
            edge_color="black",
            face_color="label",
            face_colormap=colormap_for_display,
            face_contrast_limits=(0, 1),
            edge_width=0.1,
            symbol="disc",
            size=5,
        )
        self._filter_particle_by_confidence()
        self.is_prediction = True

    def _reset_session(self):
        self.last_view = ""
        self.img_process = None
        self.tm_scores = None
        self.tm_idx = None

        self.image_name = ""
        self.filename = None
        self.tm_list = None
        self.img_process, self.patch_corner = None, None

        self.delta = None
        self.delta_values = []
        self.delta_plot.update_plot(y_values=self.delta_values)

        # Particles selections
        self.cur_proposal_index, self.proposals = 0, []
        self.user_annotations = np.zeros((0, 4))  # Z, Y, X, Label
        self.selected_particle_id = None

        # Remove after testing
        self.particle = None
        self.confidence = None
        self.patch_points, self.patch_label = np.zeros((0, 3)), np.zeros((1,))

        # BLR model
        self.model, self.model_pred, self.weights, self.bias = None, None, None, None
        self.init, self.init_done, self.AL, self.Predict = False, False, False, False
        self.AL_weights = None

        self._reset_views()
        self.clean_viewer()

    def _soft_reset_session(self):
        self.tm_idx = None
        self.delta = None
        self.patch_corner = None

        # Particles selections
        self.cur_proposal_index, self.proposals = 0, []
        self.user_annotations = np.zeros((0, 4))  # Z, Y, X, Label
        self.selected_particle_id = None

        self.delta_values = []
        self.delta_plot.update_plot(y_values=self.delta_values)

        # Remove after testing
        self.particle = None
        self.confidence = None
        self.patch_points, self.patch_label = np.zeros((0, 3)), np.zeros((1,))

        # BLR model
        self.model, self.model_pred, self.weights, self.bias = None, None, None, None
        self.init, self.init_done, self.AL, self.Predict = False, False, False, False
        self.AL_weights = None

    """""" """""" """""" """
    Viewer functionality
    """ """""" """""" """"""

    def _pdb_id_update(self):
        try:
            if self.pdb_id.value == "6QS9":
                tardis_ = [1 if i.startswith("tardis") else 0 for i in self.tm_list]

                if sum(tardis_) > 0:
                    self.tm_idx = [
                        id_
                        for id_, i in enumerate(self.tm_list)
                        if i == "tardis_" + self.pdb_id.value
                    ][0]
                else:
                    self.tm_idx = [
                        id_
                        for id_, i in enumerate(self.tm_list)
                        if i == self.pdb_id.value
                    ][0]
            else:
                self.tm_idx = [
                    id_ for id_, i in enumerate(self.tm_list) if i == self.pdb_id.value
                ][0]
        except:
            return

        if self.init:
            self.delta_values.append(0.0)
            filter_size = int(self.filter_size.value)
            # Restart user annotation storage
            self.user_annotations = np.zeros((0, 4))

            peaks, _ = find_peaks(
                self.tm_scores[self.tm_idx], filter_size, with_score=True
            )
            peaks = peaks[-20:, :]
            peaks = np.hstack((peaks, np.zeros((20, 1))))
            peaks[:, 3] = 2

            peaks_ice, _ = find_peaks(self.tm_scores[-1], filter_size, with_score=True)

            peaks_ice = peaks_ice[-10:, :]
            peaks_ice = np.hstack((peaks_ice, np.zeros((10, 1))))
            peaks_ice[:, 3] = 2

            peaks = np.vstack((peaks, peaks_ice))

            self.user_annotations = peaks
            self.patch_points = peaks[:, :3]
            self.patch_label = peaks[:, 3]
            self.patch_corner = (0, 0, 0)
            self._show_active_learning_grid()

            self.activate_click = True
        elif self.AL or self.Predict:
            self._soft_reset_session()
            if self.last_view is not None:
                if self.last_view == "Tomo":
                    self._show_tomogram()
            elif self.last_view == "AL":
                self._show_active_learning_grid()
            else:
                self._show_particle_grid()

            self.activate_click = True

    def _reset_views(self):
        self.last_view = None
        self.peaks_full = None
        self.peaks_confidence_full = None
        self.logits_full = None
        self.logits_patch = None

        (
            self.crop_grid_img_al,
            self.crop_grid_tm_scores_al,
            self.grid_particle_points_al,
            self.grid_particle_labels_al,
        ) = (
            None,
            None,
            None,
            None,
        )

        (
            self.crop_grid_img,
            self.crop_grid_tm_scores,
            self.grid_particle_points,
            self.grid_particle_labels,
        ) = (
            None,
            None,
            None,
            None,
        )

        self.is_prediction = False
        self.napari_viewer.reset_view()

    def _show_tomogram(self):
        """
        Viewer function to display a tomogram and all particles in it.
        """
        self.last_view = "Tomo"
        self.all_grid = False
        self.grid_labeling_mode = False
        self.clean_viewer()

        patch_size = int(self.patch_size.value)
        shape_ = self.img_process.shape
        gauss_filter = float(self.gauss.value)
        filter_size = int(self.filter_size.value)

        if self.patch_corner is not None:
            all_vertices = np.array(
                [
                    [
                        self.patch_corner[0],
                        self.patch_corner[1],
                        self.patch_corner[2],
                    ],
                    [
                        self.patch_corner[0],
                        self.patch_corner[1],
                        min(self.patch_corner[2] + patch_size, shape_[2]),
                    ],
                    [
                        self.patch_corner[0],
                        min(self.patch_corner[1] + patch_size, shape_[1]),
                        self.patch_corner[2],
                    ],
                    [
                        self.patch_corner[0],
                        min(self.patch_corner[1] + patch_size, shape_[1]),
                        min(self.patch_corner[2] + patch_size, shape_[2]),
                    ],
                    [
                        min(self.patch_corner[0] + patch_size, shape_[0]),
                        self.patch_corner[1],
                        self.patch_corner[2],
                    ],
                    [
                        min(self.patch_corner[0] + patch_size, shape_[0]),
                        self.patch_corner[1],
                        min(self.patch_corner[2] + patch_size, shape_[2]),
                    ],
                    [
                        min(self.patch_corner[0] + patch_size, shape_[0]),
                        min(self.patch_corner[1] + patch_size, shape_[1]),
                        self.patch_corner[2],
                    ],
                    [
                        min(self.patch_corner[0] + patch_size, shape_[0]),
                        min(self.patch_corner[1] + patch_size, shape_[1]),
                        min(self.patch_corner[2] + patch_size, shape_[2]),
                    ],
                ]
            ).squeeze()

        if self.logits_full is None:
            if self.model is not None:
                peaks, peaks_confidence, self.logits_full = predict_3d_with_AL(
                    self.img_process,
                    tm_scores=self.tm_scores,
                    model=self.model,
                    offset=patch_size,
                    maximum_filter_size=filter_size,
                    gauss_filter=gauss_filter,
                )
                order = np.argsort(peaks_confidence)
                peaks = peaks[order]
                peaks_confidence = peaks_confidence[order]

                self.peaks_full = peaks[-100:, :]
                self.peaks_confidence_full = peaks_confidence[-100:]

        self.create_image_layer(self.img_process, name="Tomogram_Patch")

        self.create_image_layer(
            self.tm_scores[self.tm_idx],
            name="TM_Scores",
            transparency=True,
            visibility=False,
        )

        try:
            self.create_image_layer(
                self.y.detach().numpy(), name="y", transparency=True, visibility=False
            )
        except:
            pass

        if self.logits_full is not None:
            self.create_image_layer(
                self.logits_full, name="Sigmoid", transparency=True, visibility=False
            )

        if self.patch_corner is not None:
            if self.patch_corner != (0, 0, 0):
                bb_layer = BoundingBoxLayer(ndim=3, edge_color="red", edge_width=5)
                bb_layer.add(all_vertices)
                self.napari_viewer.add_layer(bb_layer)

        if self.model is not None:
            self.napari_viewer.add_points(
                self.peaks_full,
                name="Particle_top_100",
                properties={"label": self.peaks_confidence_full},
                edge_color="black",
                face_color="label",
                face_colormap=colormap_for_display,
                face_contrast_limits=(0, 1),
                edge_width=0.1,
                symbol="disc",
                size=5,
            )

        self.create_point_layer(
            self.user_annotations[:, :3],
            self.user_annotations[:, 3],
            name="All_labels",
            visible=False,
        )

        if self.is_prediction:
            self.napari_viewer.add_points(
                self.peaks_full,
                name="Particle_Prediction",
                properties={"label": self.peaks_confidence_full},
                edge_color="black",
                face_color="label",
                face_colormap=colormap_for_display,
                face_contrast_limits=(0, 1),
                edge_width=0.1,
                symbol="disc",
                size=5,
            )

            self._filter_particle_by_confidence()
        else:
            if self.patch_points is not None and self.logits_full is not None:
                if self.patch_corner is not None:
                    corrected_points = correct_coord(
                        self.patch_points.copy(), self.patch_corner, True
                    )
                else:
                    corrected_points = self.patch_points.copy()

                self.create_point_layer(
                    corrected_points,
                    self.patch_label,
                    name="Particle_BLR_is_Uncertain",
                )
            else:
                self.create_point_layer(
                    self.user_annotations[:, :3],
                    self.user_annotations[:, 3],
                    name="Initial_Particle_Selection",
                )
        self.napari_viewer.reset_view()

    def _show_active_learning_grid(self):
        """
        Viewer function to show all stored particles in current patch based on
        self.user_annotation and particle_layer. Particles are shown as a grid
        with particle in the center.
        """
        self.last_view = "AL"
        if self.patch_corner is not None:
            self.all_grid = False
            self.grid_labeling_mode = True
            self.clean_viewer()

            (
                self.crop_grid_img_al,
                self.crop_grid_tm_scores_al,
                self.grid_particle_points_al,
                self.grid_particle_labels_al,
            ) = build_gird_with_particles(
                self.patch_points,
                self.patch_label,
                self.patch_corner,
                self.img_process,
                (
                    self.tm_scores
                    if self.logits_patch is None
                    else self.logits_patch[None, ...]
                ),
                self.tm_idx if self.logits_patch is None else 0,
                int(self.box_size.value),
                correct=True if self.logits_patch is None else False,
            )

            self.create_image_layer(self.crop_grid_img_al, name="Particles_crops")
            if self.logits_patch is None:
                self.create_image_layer(
                    self.crop_grid_tm_scores_al,
                    name="Particles_crops_scores",
                    transparency=True,
                    visibility=False,
                    range_=None,
                )
            else:
                self.create_image_layer(
                    self.crop_grid_tm_scores_al,
                    name="Particles_crops_sigmoid",
                    transparency=True,
                    visibility=False,
                    range_=(0, 1),
                )

            self.create_point_layer(
                self.grid_particle_points_al,
                self.grid_particle_labels_al,
                "Particle_BLR_is_Uncertain",
            )
        else:
            return

    def _show_particle_grid(self):
        """
        Viewer function to show all stored particles on self.user_annotation
        and particle_layer. Particles are shown as a grid with particle in the center.
        """
        self.last_view = "Grid"
        self.all_grid = True
        self.grid_labeling_mode = True

        if self.is_prediction:
            particles_filter_predict = self.napari_viewer.layers[
                "Particle_Prediction_Filtered"
            ].data
            labels_filter_predict = self.napari_viewer.layers[
                "Particle_Prediction_Filtered"
            ].properties["label"]

            (
                self.crop_grid_img,
                self.crop_grid_tm_scores,
                self.grid_particle_points,
                self.grid_particle_labels,
            ) = build_gird_with_particles(
                particles_filter_predict,
                labels_filter_predict,
                (0, 0, 0),
                self.img_process,
                (
                    self.tm_scores
                    if self.logits_full is None
                    else self.logits_full[None, ...]
                ),
                self.tm_idx if self.logits_full is None else 0,
                int(self.box_size.value),
                correct=False,
            )
        else:
            (
                self.crop_grid_img,
                self.crop_grid_tm_scores,
                self.grid_particle_points,
                self.grid_particle_labels,
            ) = build_gird_with_particles(
                self.user_annotations[:, :3],
                self.user_annotations[:, 3],
                (0, 0, 0),
                self.img_process,
                (
                    self.tm_scores
                    if self.logits_full is None
                    else self.logits_full[None, ...]
                ),
                self.tm_idx if self.logits_full is None else 0,
                int(self.box_size.value),
                correct=False,
            )

        self.clean_viewer()
        self.create_image_layer(self.crop_grid_img, name="Particles_crops")
        self.create_image_layer(
            self.crop_grid_tm_scores,
            name="Particles_crops_scores",
            transparency=True,
            visibility=False,
            range_=(0, 1) if self.logits_full is not None else None,
        )

        if self.is_prediction:
            self.napari_viewer.add_points(
                self.grid_particle_points,
                name="Particle_Prediction",
                properties={"label": self.grid_particle_labels},
                edge_color="black",
                face_color="label",
                face_colormap=colormap_for_display,
                face_contrast_limits=(0, 1),
                edge_width=0.1,
                symbol="disc",
                size=5,
            )
        else:
            self.create_point_layer(
                self.grid_particle_points,
                self.grid_particle_labels,
                "Particle_BLR_is_Uncertain",
            )

    """""" """""" """""" """""" """
    Viewer helper functionality
    """ """""" """""" """""" """"""

    def clean_viewer(self):
        self.napari_viewer.layers.select_all()
        self.napari_viewer.layers.remove_selected()

    def create_image_layer(
        self, image, name="TM_Scores", transparency=False, visibility=True, range_=None
    ):
        """
        Create a image layer in napari.

        Args:
            tm_scores (np.ndarray): Image array to display
            name (str): Layer name
            transparency (bool): If True, show image as transparent layer
        """
        try:
            self.napari_viewer.layers.remove(name)
        except Exception as e:
            pass

        if transparency:
            self.napari_viewer.add_image(
                image, name=name, colormap=colormap_for_display, opacity=0.5
            )
        else:
            self.napari_viewer.add_image(image, name=name, colormap="gray", opacity=1.0)

        if range_ is not None:
            try:
                self.napari_viewer.layers[name].contrast_limits = (
                    range_[0],
                    range_[1],
                )
            except Exception as e:
                pass
        else:
            try:
                self.napari_viewer.layers[name].contrast_limits = (
                    image.min(),
                    image.max(),
                )
            except Exception as e:
                pass

        if visibility:
            # set layer as not visible
            self.napari_viewer.layers[name].visible = True
        else:
            self.napari_viewer.layers[name].visible = False

    def create_point_layer(
        self, point: np.ndarray, label: np.ndarray, name="Initial_Labels", visible=True
    ):
        """
        Create a point layer in napari with 2D/3D points and associated labels.
        """

        # If layer of the same exist remove it for update
        # Overwriting layer results in not correctly displayed points labels
        try:
            self.napari_viewer.layers.remove(name)
        except Exception as e:
            pass

        if point.shape[0] > 0:
            self.napari_viewer.add_points(
                point,
                name=name,
                face_color="#00000000",  # Hex + alpha
                properties={"label": label.astype(np.int16)},
                edge_color="label",
                edge_color_cycle=self.color_map_particle_classes,
                edge_width=0.1,
                face_contrast_limits=(0, 2),
                symbol="square",
                size=40,
            )
        else:  # Create empty layer
            self.napari_viewer.add_points(
                point,
                name=name,
                face_color="#00000000",  # Hex + alpha
                edge_color_cycle=self.color_map_particle_classes,
                edge_width=0.1,
                face_contrast_limits=(0, 2),
                symbol="square",
                size=40,
            )

        self.napari_viewer.layers[name].mode = "select"
        self.napari_viewer.layers[name].visible = visible

    def _filter_particle_by_confidence(self):
        """
        Function to fetch
        self.napari_viewer.layers.selection.active.name["Prediction_Filtered"]
        and filter particle based on the confidence scored given from
        self.filter_particle_by_confidence.value

        Function updated ..._Prediction_Filtered Points layer.
        """
        try:
            self.napari_viewer.layers["Particle_Prediction"].visible = False
            particles_all = self.napari_viewer.layers["Particle_Prediction"].data
            confidence_all = self.napari_viewer.layers[
                "Particle_Prediction"
            ].properties["label"]

            if len(particles_all) == 0:
                show_info("No predicted particles to filter!")
                return

            keep_id = np.where(
                confidence_all >= self.filter_particle_by_confidence.value
            )
            particles_filter = particles_all[keep_id[0], :]
            confidence_filter = confidence_all[keep_id[0]]

            try:
                self.napari_viewer.layers.remove("Particle_Prediction_Filtered")
            except:
                pass

            if len(particles_filter) > 0:
                self.napari_viewer.add_points(
                    particles_filter,
                    name="Particle_Prediction_Filtered",
                    properties={"label": confidence_filter},
                    edge_color="black",
                    face_color="label",
                    face_colormap=colormap_for_display,
                    edge_width=0.1,
                    face_contrast_limits=(0, 1),
                    symbol="disc",
                    size=5,
                )
        except Exception as e:
            pass

    """""" """""" """""" """""
    Global helper functions
    """ """""" """""" """""" ""

    def _export(self, data: np.ndarray):
        """
        General export function to .star file format
        """
        # Save only user annotations (positive labels)
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File", directory="user_annotations.csv"
        )

        if filename.endswith(".star"):
            filename = filename.split(".")
            filename = filename[0] + ".csv"

        # Legacy file
        np.savetxt(
            filename, data, delimiter=",", fmt="%s", header="Z, Y, X, Confidence"
        )

        filename = filename[:-3] + "star"
        data = pd.DataFrame(
            data=data,
            columns=[
                "_rlnConfidence",
                "_rlnCoordinateZ",
                "_rlnCoordinateY",
                "_rlnCoordinateX",
            ],
        )
        starfile.write(data, filename)

    def _export_particles_all(self):
        user_annotations = self.user_annotations.copy()

        try:
            prediction_particle = self.napari_viewer.layers["Particle_Prediction"].data
            prediction_labels = self.napari_viewer.layers[
                "Particle_Prediction"
            ].properties["label"]
        except:
            prediction_particle, prediction_labels = [], []

        if len(prediction_particle) == 0:
            data = user_annotations
        else:
            prediction = np.hstack((prediction_particle, prediction_labels[:, None]))
            min_, max_ = np.min(prediction_labels), np.max(prediction_labels)

        user_annotations[user_annotations[:, 3] == 0, 3] = min_
        user_annotations[user_annotations[:, 3] == 1, 3] = max_
        data = np.concatenate((user_annotations, prediction))

        self._export(data)

    def _export_particles_labeled(self):
        self._export(self.user_annotations.copy())

    def _export_particles_predict(self):
        try:
            prediction_particle = self.napari_viewer.layers["Particle_Prediction"].data
            prediction_labels = self.napari_viewer.layers[
                "Particle_Prediction"
            ].properties["label"]
        except:
            prediction_particle, prediction_labels = [], []

        prediction = np.hstack((prediction_particle, prediction_labels[:, None]))

        self._export(prediction)

    def _export_particles_filter(self):
        try:
            prediction_particle = self.napari_viewer.layers[
                "Particle_Prediction_Filtered"
            ].data
            prediction_labels = self.napari_viewer.layers[
                "Particle_Prediction_Filtered"
            ].properties["label"]
        except:
            prediction_particle, prediction_labels = [], []

        prediction = np.hstack((prediction_particle, prediction_labels[:, None]))

        self._export(prediction)

    def _export_particles(self):
        """
        Fetch all positive and negative particle, and export it as .csv file
        with header [Z, Y, X, Score].

        Fetched points should be from all already labels by user or predicted.
        If positive is present score is 1 or max. confidence score from prediction
        if present. For negative prediction it should be -1 or min. confidence
        score from prediction if present.

        Export self.user_annotations [n, 4] organized Z, Y, X, ID
        """
        user_annotations = self.user_annotations.copy()

        try:
            prediction_particle = self.napari_viewer.layers["Particle_Prediction"].data
            prediction_labels = self.napari_viewer.layers[
                "Particle_Prediction"
            ].properties["label"]
        except:
            prediction_particle, prediction_labels = [], []

        # Save only user annotations (positive labels)
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File", directory="user_annotations.csv"
        )

        if len(prediction_particle) == 0:
            data = user_annotations
        else:
            prediction = np.hstack((prediction_particle, prediction_labels[:, None]))
            min_, max_ = np.min(prediction_labels), np.max(prediction_labels)

        user_annotations[user_annotations[:, 3] == 0, 3] = min_
        user_annotations[user_annotations[:, 3] == 1, 3] = max_
        data = np.concatenate((user_annotations, prediction))

        np.savetxt(
            filename, data, delimiter=",", fmt="%s", header="Z, Y, X, Confidence"
        )

    def _import_particles(self):
        """
        Import file with coordinates. Expect that files contains point in XYZ order,
        with optional confidence scores.
        Use "viridis" colormap them for the scores. If score are not present,
        assign all with score 0.

        Allow for [n, 3] or [n, 4]
        if napari binder save
        df = [n, 3 or 4] read it as [1:, 1:] [ZYX]
        """
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        data, labels = load_coordinates(self.filename)

        # try:
        # Update user annotation storage
        self.user_annotations = np.concatenate(
            (self.user_annotations, np.hstack((data, labels[:, None])))
        )
        self.user_annotations = np.vstack(tuple(set(map(tuple, self.user_annotations))))

        # add imported points to the layer
        try:
            self.napari_viewer.layers.remove("Imported_Particles")
        except:
            pass

        self.napari_viewer.add_points(
            data,
            name="Imported_Particles",
            properties={"label": labels},
            edge_color="black",
            face_color="label",
            face_colormap=colormap_for_display,
            edge_width=0.1,
            symbol="disc",
            size=5,
        )
        show_info(f"Imported {data.shape[0]} particles!")

    def _save_model(self):
        """
        Function to fetch self.AL_weights which is a list [self.weight, self.bias]
        and save it as a pickle torch .pt file
        """
        if self.model is not None:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save File", directory="Active_learn_model.pt"
            )

            # Check for AL_weights
            if (
                self.AL_weights is None
            ):  # Hard-fix in case self.AL_weights is not save yet
                self.AL_weights = [self.model.weights, self.model.bias]

            torch.save(self.AL_weights, filename)

    def _load_model(self):
        """
        Function to load and update self.AL_weights which is a list [self.weight, self.bias]
        expected as a pickle torch .pt file.

        If self.model is not None, update model weights. Else create self.model with
        this weights.
        """
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        self.AL_weights = torch.load(f"{self.filename}")

        if self.model is not None:
            self.model.fit(pre_train=self.AL_weights)
        else:
            self.model = BinaryLogisticRegression(
                n_features=self.AL_weights[0].shape[0],
                l2=1.0,
                pi=float(self.pi.value),
                pi_weight=1000,
            )
            self.model.fit(pre_train=self.AL_weights)

        self.AL_weights = None  # Reset to allow re-training

    """""" """""" """""" """""
    Mouse and keys bindings
    """ """""" """""" """""" ""

    def track_mouse_position(self, viewer: Viewer, event):
        """
        Mouse binding helper function to update stored mouse position when it moves
        """
        self.mouse_position = event.position

    def selected_point_near_mouse(self, viewer: Viewer, event):
        """
        Mouse binding helper function to select point near the mouse pointer
        """
        if self.activate_click:
            name = viewer.layers.selection.active.name

            try:
                # if self.activate_click:
                points_layer = viewer.layers[name].data

                # Filter points_layer and search only for points withing radius
                # Just in case we have thousands or millions of points issue
                points_layer = points_layer[
                    np.where(
                        np.linalg.norm(points_layer - self.mouse_position, axis=1) < 10
                    )
                ]

                # Calculate the distance between the mouse position and all points
                distances = np.linalg.norm(points_layer - self.mouse_position, axis=1)
                closest_point_index = distances.argmin()

                # Clear the current selection and Select the closest point
                if self.selected_particle_id != closest_point_index:
                    self.selected_particle_id = closest_point_index

                    viewer.layers[name].selected_data = set()
                    viewer.layers[name].selected_data.add(closest_point_index)
            except Exception as e:
                pass

    def update_point_grid(self, index=None, label=0, func="update"):
        point_layer = self.napari_viewer.layers["Particle_BLR_is_Uncertain"]
        points = point_layer.data
        labels = point_layer.properties["label"]

        if self.all_grid:
            point = self.user_annotations[index]
        else:
            point = self.patch_points[index]

        if func == "update":
            # Update point labels
            if labels[index] != label:
                labels[index] = label

            if self.all_grid:
                self.user_annotations[:, 3] = labels
            else:
                self.patch_label = labels

            # Check if point index from self.point_layer is already in self.user_annotations
            # Add and/or update point label in self.user_annotations
            if not self.all_grid:
                point = correct_coord(point, self.patch_corner, True)

                idx = self.user_annotations[:, :3] - point
                idx_bool = 0 in np.sum(idx.astype(np.float16), axis=1)

                if idx_bool:  # Add point to self.user_annotation
                    idx = np.where(np.sum(idx.astype(np.float16), axis=1) == 0)

                    if len(idx) > 0:
                        self.user_annotations[idx[0][0], 3] = label
                else:  # Update point label in self.user_annotations
                    self.user_annotations = np.insert(
                        self.user_annotations,
                        0,
                        np.insert(point, 3, [label], axis=1),
                        axis=0,
                    )
        elif func == "remove":
            if self.all_grid:
                self.user_annotations = np.delete(
                    self.user_annotations[:, :3], index, axis=0
                )
                self.user_annotations[:, 3] = np.delete(
                    self.user_annotations[:, 3], index, axis=0
                )
                points = np.delete(self.user_annotations[:, :3], index, axis=0)
                labels = np.delete(self.user_annotations[:, 3], index, axis=0)
            else:
                self.patch_points = np.delete(self.patch_points, index, axis=0)
                self.patch_label = np.delete(self.patch_label, index, axis=0)
                points = np.delete(self.patch_points, index, axis=0)
                labels = np.delete(self.patch_points, index, axis=0)

                idx = points in self.user_annotations[:, :3]
                if idx:  # Remove point to self.user_annotation
                    idx = np.where(points in self.user_annotations[:, :3])[0][0]

                    self.user_annotations = np.delete(
                        self.user_annotations, index, axis=0
                    )

        self.create_point_layer(points, labels, "Particle_BLR_is_Uncertain")

    def update_point_layer(self, index=None, label=0, func="add"):
        name = self.napari_viewer.layers.selection.active.name
        point_layer = self.napari_viewer.layers[name]

        # try:
        #     point_layer = self.napari_viewer.layers["Particle_BLR_is_Uncertain"]
        #     # self.patch_points = point_layer.data
        #     # self.patch_label = point_layer.properties["label"]
        # except Exception as e:
        #     pass

        # Add point pointed by mouse
        if func == "add":
            # Add point
            points = point_layer.data

            if points.shape[0] == 0:
                labels = np.array([label])
                points = np.array([self.mouse_position])
            else:
                labels = point_layer.properties["label"]
                labels = np.insert(labels, len(points), [label], axis=0)

                points = np.insert(points, len(points), self.mouse_position, axis=0)

            # Update user annotation storage
            if self.correct_positions:
                self.user_annotations = np.concatenate(
                    (
                        self.user_annotations,
                        np.hstack(
                            (
                                correct_coord(
                                    np.array([self.mouse_position]),
                                    self.patch_corner,
                                    True,
                                ),
                                np.array([label])[:, None],
                            )
                        ),
                    )
                )
            else:
                self.user_annotations = np.concatenate(
                    (
                        self.user_annotations,
                        np.hstack(
                            (
                                np.array([self.mouse_position]),
                                np.array([label])[:, None],
                            )
                        ),
                    )
                )
            self.user_annotations = np.vstack(
                tuple(set(map(tuple, self.user_annotations)))
            )
        elif func == "remove":  # Remove point pointed by mouse
            points = point_layer.data
            labels = point_layer.properties["label"]

            # Remove point from user annotation storage
            if self.correct_positions:
                idx = np.where(
                    np.all(
                        self.user_annotations[:, :3]
                        == correct_coord(points[index], self.patch_corner, True),
                        axis=1,
                    )
                )
            else:
                idx = np.where(
                    np.all(self.user_annotations[:, :3] == points[index], axis=1)
                )

            if len(idx) == 0:
                show_info("No matching point in index to remove.")
            else:
                self.user_annotations = np.delete(self.user_annotations, idx, axis=0)

            # Remove point from layer
            points = np.delete(points, index, axis=0)
            labels = np.delete(labels, index, axis=0)
        elif func == "update":  # Update point pointed by mouse
            points = point_layer.data
            labels = point_layer.properties["label"]

            # Update point in user annotation storage
            if self.correct_positions:
                idx = np.where(
                    np.all(
                        self.user_annotations[:, :3]
                        == correct_coord(points[index], self.patch_corner, True),
                        axis=1,
                    )
                )[0]
            else:
                idx = np.where(
                    np.all(self.user_annotations[:, :3] == points[index], axis=1)
                )[0]

            if len(idx) == 0:
                if self.correct_positions:
                    self.user_annotations = np.concatenate(
                        (
                            self.user_annotations,
                            np.hstack(
                                (
                                    correct_coord(
                                        points[index], self.patch_corner, True
                                    ),
                                    np.array([label])[:, None],
                                )
                            ),
                        )
                    )
                else:
                    self.user_annotations = np.concatenate(
                        (
                            self.user_annotations,
                            np.hstack((points[index], np.array([label])[:, None])),
                        )
                    )
                self.user_annotations = np.vstack(
                    tuple(set(map(tuple, self.user_annotations)))
                )
            else:
                if self.user_annotations[idx[0], -1] != label:
                    self.user_annotations[idx[0], -1] = label

            # Update point labels
            if labels[index] != label:
                labels[index] = label

        # Update point layer
        self.create_point_layer(points, labels, name)
        point_layer.edge_color_cycle = self.color_map_particle_classes

    def key_event(self, viewer: Viewer, key: int):
        """
        Main key event definition.

        Args:
            key (int): Key definition of an action.
        """
        if self.activate_click:
            name = viewer.layers.selection.active.name
            points_layer = viewer.layers[name].data

            mouse_position = np.asarray(self.mouse_position).reshape(1, -1)
            if points_layer.shape[0] == 0:
                self.update_point_layer(None, key, "add")
            else:
                kdtree = KDTree(points_layer)
                distance, closest_point = kdtree.query(mouse_position, k=1)

                if not self.grid_labeling_mode:
                    if key == 2:
                        if distance[0] < 10:
                            self.update_point_layer(closest_point[0], 0, "remove")
                    else:
                        if distance[0] > 10:
                            self.update_point_layer(None, key, "add")
                        else:
                            self.update_point_layer(closest_point[0], key, "update")
                else:
                    if key in [0, 1]:
                        self.update_point_grid(closest_point[0], key, "update")
                    else:
                        self.update_point_grid(closest_point[0], key, "remove")

    def ZEvent(self, viewer):
        self.key_event(viewer, 0)

    def XEvent(self, viewer):
        self.key_event(viewer, 1)

    def CEvent(self, viewer):
        self.key_event(viewer, 2)


class PlotPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Napari Active Learning training progress")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        plt.style.use(["dark_background"])

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def update_plot(self, y_values):
        y_values = [round(value, 3) for value in y_values]

        x_values = np.arange(
            len(y_values)
        )  # Automatically generate x-axis values based on length of y_values

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x_values, y_values)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Score")
        self.canvas.draw()
