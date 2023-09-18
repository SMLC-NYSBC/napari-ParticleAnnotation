from enum import Enum
from functools import partial
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
    Label,
    FloatSlider,
    Checkbox,
)
from napari import Viewer
from scipy.spatial import KDTree
from topaz.stats import normalize
from vispy.geometry import Rect

from napari.components import ViewerModel
from copy import deepcopy

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter, QFileDialog

from packaging.version import parse as parse_version

import napari
from scipy.ndimage import maximum_filter
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Points, Layer
from napari.qt import QtViewer
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info

from napari_ParticleAnnotation.utils.model.active_learning_model import (
    BinaryLogisticRegression,
    initialize_model,
    label_points_to_mask,
)

from napari_ParticleAnnotation.utils.load_data import downsample
from napari_ParticleAnnotation.utils.model.utils import rank_candidate_locations

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


class SearchType(Enum):
    Highlight = 0
    Zoom_in = 1


def copy_layer_le_4_16(layer: Layer, name: str = ""):
    """
    Create a copy of the layer and sets the viewer name. The data of the copied layer is shared with the original layer.

    Args:
        layer (Union[Labels, Image]): The layer to copy.
        name (Optional, str): The viewer name to set for the copied layer.

    Returns:
        Union[Labels, Image]: A copy of the layer.

    Notes:
        - If the layer is an Image or Labels layer, the data is not deeply copied.
        - This function assumes that the layer has a `metadata` attribute and an `events` attribute.
        - This function is optimized for napari version 0.4.16 or earlier.
    """
    # Shallow copy of the layer object
    res_layer = deepcopy(layer)

    # If the layer is an Image or Labels layer, share the data with the original layer
    if isinstance(layer, (Image, Points)):
        res_layer.data = layer.data

    # Set the viewer name for the copied layer
    res_layer.metadata["viewer_name"] = name

    # Disconnect the original layer's events and connect the copied layer's events
    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer

    return res_layer


def copy_layer(layer: Layer, name: str = ""):
    # If napari version < 0.4.16, create a new layer object with the same data as the original layer
    res_layer = Layer.create(*layer.as_layer_data_tuple())

    # Share the data with the original layer if it's an Image or Labels layer
    if isinstance(layer, (Image, Points)):
        res_layer.data = layer.data

    # Set the viewer name for the copied layer
    res_layer.metadata["viewer_name"] = name

    return res_layer


def get_property_names(layer: Layer):
    class_ = layer.__class__
    res = []

    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue

        # Check if the event_name attribute is a property with a setter method
        if (
            isinstance(getattr(class_, event_name, None), property)
            and getattr(class_, event_name).fset is not None
        ):
            res.append(event_name)

    return res


def center_cross_on_mouse(
    viewer_model: napari.components.viewer_model.ViewerModel,
):
    """move the cross to the mouse position"""

    if not getattr(viewer_model, "mouse_over_canvas", True):
        # Notify the user that the mouse is not over the viewer canvas
        show_info("Mouse is not over the canvas. You may need to click on the canvas.")
        return

    viewer_model.dims.current_step = tuple(
        np.round(
            [
                max(min_, min(p, max_)) / step
                for p, (min_, max_, step) in zip(
                    viewer_model.cursor.position, viewer_model.dims.range
                )
            ]
        ).astype(int)
    )


action_manager.register_action(
    name="napari:move_point",
    command=center_cross_on_mouse,
    description="Move dims point to mouse position",
    keymapprovider=ViewerModel,
)

action_manager.bind_shortcut("napari:move_point", "M")
action_manager.bind_shortcut("napari:add_point", "W")
action_manager.bind_shortcut("napari:add_positive", "A")
action_manager.bind_shortcut("napari:add_negative", "D")
action_manager.bind_shortcut("napari:remove", "S")


class OwnPartial:
    """
    A workaround for deepcopy not copying functools.partial objects.

    This class wraps a partial function, providing a __deepcopy__ method
    that returns a new instance of the wrapped function with the same
    arguments and keyword arguments.

    Note that this class is intended to be used as a last resort and should
    only be used when it is not possible to serialize the original object
    directly.
    """

    def __init__(self, func, *args, **kwargs):
        self.func = partial(func, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __deepcopy__(self, memodict={}):
        return OwnPartial(
            self.func.func,
            *deepcopy(self.func.args, memodict),
            **deepcopy(self.func.keywords, memodict),
        )


class QtViewerWrap(QtViewer):
    """
    A wrapper around the QtViewer class that provides drag-and-drop file
    opening functionality.

    This class stores a reference to a main viewer object and overrides the
    _qt_open method of the QtViewer class to open files in the main viewer.

    Args:
         main_viewer (Viewer): he main viewer object to which files should be opened.
        *args, **kwargs (Any): Additional arguments and keyword arguments to pass to the QtViewer
            constructor.
    """

    def __init__(self, main_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str = None,
        layer_type: str = None,
        **kwargs,
    ):
        """
        Open the specified files in the main viewer object.

        Args:
            filenames (list[str]): The list of filenames to open.
            stack (bool): Whether to stack the images as layers.
            plugin (optional, str): The name of the plugin to use for opening the files (default is None).
            layer_type (optional, str): The layer type to use for opening the files (default is None).
            **kwargs (any): Additional keyword arguments to pass to the QtViewer _qt_open method.
        """
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

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

        self.annotation_widget.reset_view.clicked.connect(self._reset_view)

        self.viewer.camera.events.connect(self._sync_view)

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

    def _reset_view(self):
        self.viewer.reset_view()
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _sync_view(self):
        self.viewer_model2.camera.zoom = self.viewer.camera.zoom

        layer_index = self.viewer_model1.layers.index(self.points_layer)
        self.viewer_model1.layers.move(layer_index, -1)

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

        self.viewer_model2.camera.center = (points[0], points[1], points[2])

        self.viewer_model2.dims.set_point(-1, points[2])

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

        self.viewer_model1.layers.selection.active = self.viewer_model1.layers[
            event.value.name
        ]
        self.viewer_model2.layers.selection.active = self.viewer_model2.layers[
            event.value.name
        ]

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            return

        # order[-3:] = order[-2], order[-3], order[-1]
        # self.viewer_model1.dims.order = order
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model2.dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""
        self.viewer_model1.layers.insert(event.index, copy_layer(event.value, "View_1"))
        self.viewer_model2.layers.insert(event.index, copy_layer(event.value, "View_2"))
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                OwnPartial(self._property_sync, name)
            )

        if isinstance(event.value, Points):
            event.value.events.set_data.connect(self._set_data_refresh)
            self.viewer_model1.layers[event.value.name].events.set_data.connect(
                self._set_data_refresh
            )
            self.viewer_model2.layers[event.value.name].events.set_data.connect(
                self._set_data_refresh
            )

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
        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
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
            setattr(
                self.viewer_model1.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
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

        self.click_add_point_callback = None
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

        # Key binding
        try:
            self.napari_viewer.bind_key("z", self.ZEvent)
            self.napari_viewer.bind_key("x", self.XEvent)
            self.napari_viewer.bind_key("c", self.CEvent)
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

        # Initialize model
        self.load_ALM = PushButton(name="Load model")
        self.load_ALM.clicked.connect(self._load_model)
        self.save_ALM = PushButton(name="Save model")
        self.save_ALM.clicked.connect(self._save_model)

        spacer1 = Label(value="------- Initialize New Dataset ------")
        self.sampling_layer = LineEdit(name="Pixel_size", value="1.0")
        self.box_size = LineEdit(name="Box size", value="5")

        self.init_data = PushButton(name="Initialize dataset")
        self.init_data.clicked.connect(self._initialize_dataset)

        self.recenter_positive = Checkbox(
            name="Recenter new positive labels", value=False
        )

        spacer2 = Label(value="------ Initialize Active learning model -------")
        self.refresh = PushButton(name="Retrain")
        self.refresh.clicked.connect(self._refresh)
        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)

        spacer3 = Label(value="------------ Visualize labels tool ------------")
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

        self.zoom_factor = create_widget(
            annotation=float, label="Zoom factor", value=100
        )
        self.zoom_factor.changed.connect(self._component_num_changed)

        self.reset_view = PushButton(name="Reset View")
        self.reset_view.clicked.connect(self._reset_view)

        spacer4 = Label(value="------------ Manual labels tool ------------")
        self.manual_label = PushButton(name="Gaussian pre-process")
        self.manual_label.clicked.connect(self.initialize_labeling)

        layer_init = VBox(
            widgets=(
                self.sampling_layer,
                self.box_size,
                self.init_data,
                self.recenter_positive,
            )
        )
        layer_AL = HBox(widgets=(self.refresh, self.predict))
        layer_slider = HBox(widgets=(self.slide_pred,))
        layer_visual1 = HBox(widgets=(self.points_layer, self.component_selector))
        layer_visual2 = HBox(widgets=(self.zoom_factor, self.reset_view))
        label = HBox(widgets=(self.manual_label,))
        # self.insert(0, layout_model)
        self.insert(1, spacer1)
        self.insert(2, layer_init)
        self.insert(3, spacer2)
        self.insert(4, layer_AL)
        self.insert(5, spacer3)
        self.insert(6, layer_slider)
        self.insert(7, layer_visual1)
        self.insert(8, layer_visual2)
        self.insert(9, spacer4)
        self.insert(10, label)

    def _load_model(self):
        """Logic to load pre-train active learning model"""
        self.filename, _ = QFileDialog.getSaveFileName(caption="Load File")
        self.AL_weights = torch.load(f"{self.filename}.pth")

    def _save_model(self):
        """Logic to save pre-train active learning model"""
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File", directory="Active_learn_model.pth"
        )
        if self.model is not None:
            torch.save([self.model.weights, self.model.bias], filename)

    def _initialize_dataset(self):
        # Image data
        active_layer_name = self.napari_viewer.layers.selection.active.name
        self.image_layer_name = active_layer_name
        img = self.napari_viewer.layers[active_layer_name]

        """Down_sample dataset"""
        factor = float(self.sampling_layer.value) / 8
        self.img_process = downsample(img.data, factor=factor)

        self.shape = self.img_process.shape
        # _min, _max = np.quantile(self.img_process.ravel(), [0.1, 0.9])
        img.data, _ = normalize(self.img_process, method="gmm", use_cuda=False)

        self.napari_viewer.layers[active_layer_name].contrast_limits = (
            img.data.min(),
            img.data.max(),
        )

        # Initialize dataset
        self.x, _, p_label = initialize_model(img.data)

        """ Initialize model and pick initial particles """
        self.create_point_layer(p_label[:, 1:], p_label[:, 0])
        self.activate_click = True

        """Initialize new model or load pre-trained"""
        # update y and count
        self.y = label_points_to_mask([], self.shape, self.box_size.value)
        self.count = torch.where(
            ~torch.isnan(self.y), torch.ones_like(self.y), torch.zeros_like(self.y)
        )

        if self.model is None:
            self.model = BinaryLogisticRegression(
                n_features=self.x.shape[1], l2=1.0, pi=0.01, pi_weight=1000
            )
        if self.filename is not None:
            self.model.fit(
                self.x,
                self.y.ravel(),
                weights=self.count.ravel(),
                pre_train=self.AL_weights,
            )
        self._reset_view()
        show_info(f"Task finished: Initialize Dataset!")

    def _refresh(self):
        """
        Re-train model, and add 10 most uncertain points to the list
        """
        self.activate_click = True

        points_layer = self.napari_viewer.layers["Initial_Labels"].data
        label = self.napari_viewer.layers["Initial_Labels"].properties["label"]

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

            self.y = label_points_to_mask(data, self.shape, self.box_size.value)
            self.count = (~torch.isnan(self.y)).float()
            self.model.fit(self.x, self.y.ravel(), weights=self.count.ravel())

            self.cur_proposal_index, self.proposals = rank_candidate_locations(
                self.model, self.x, self.shape, self.proposals, id_=1
            )

            # Add point which model are least certain about
            points = np.vstack(self.proposals[-10:])
            label_unknown = np.zeros((points.shape[0],))
            label_unknown[:] = 2

            data = np.vstack((data[:, 1:], points.astype(np.float64)))

            labels = np.hstack((label, label_unknown))
            self.create_point_layer(data, labels)

            show_info(f"Task finished: Retrain model!")

    def _predict(self):
        self.activate_click = False

        # Retrain before prediction if no model was loaded!
        points_layer = self.napari_viewer.layers["Initial_Labels"].data
        label = self.napari_viewer.layers["Initial_Labels"].properties["label"]

        if np.any(label == 2):
            show_info(f"Please Correct all uncertain particles!")
        else:
            data = np.asarray(points_layer.data)
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

            self.y = label_points_to_mask(data, self.shape, self.box_size.value)
            self.count = (~torch.isnan(self.y)).float()

            self.model.fit(self.x, self.y.ravel(), weights=self.count.ravel())

            with torch.no_grad():
                logits = self.model(self.x).reshape(*self.shape)
                p_sigm = torch.sigmoid(logits)
                p = torch.clone(logits)

            logits = p.numpy()

            max_filter = maximum_filter(logits, size=25)
            peaks = logits - max_filter
            peaks = np.where(peaks == 0)
            peaks = np.stack(peaks, axis=-1)
            if peaks.shape[1] == 3:
                peak_logits = p[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
            else:
                peak_logits = p[peaks[:, 0], peaks[:, 1]]

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
            self._reset_view()
            try:
                self.napari_viewer.layers["Initial_Labels"].visible = False
            except:
                pass
            self.slide_pred.value = 0

            self.particle = peaks
            self.confidence = peak_logits.numpy()
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
        # _min, _max = np.quantile(self.img_process.ravel(), [0.1, 0.9])
        img.data, _ = normalize(self.img_process, method="gmm", use_cuda=False)

        self.napari_viewer.layers[active_layer_name].contrast_limits = (
            img.data.min(),
            img.data.max(),
        )

        img.data = nd.gaussian_filter(img.data, 2)

    def ZEvent(self, viewer):
        if self.activate_click:
            # if self.activate_click:
            points_layer = viewer.layers["Initial_Labels"].data

            # Calculate the distance between the mouse position and all points
            kdtree = KDTree(points_layer)
            distance, closest_point_index = kdtree.query(self.mouse_position, k=1)

            if distance > 12:
                self.update_point_layer_2(self.mouse_position, 0, "add")
            else:
                self.update_point_layer_2(closest_point_index, 0, "update")

    def XEvent(self, viewer):
        if self.activate_click:
            # if self.activate_click:
            points_layer = viewer.layers["Initial_Labels"].data

            # Calculate the distance between the mouse position and all points
            kdtree = KDTree(points_layer)
            distance, closest_point_index = kdtree.query(self.mouse_position, k=1)

            if distance > 12:
                self.update_point_layer_2(self.mouse_position, 1, "add")
            else:
                self.update_point_layer_2(closest_point_index, 1, "update")

    def CEvent(self, viewer):
        if self.activate_click:
            # if self.activate_click:
            points_layer = viewer.layers["Initial_Labels"].data

            # Calculate the distance between the mouse position and all points
            kdtree = KDTree(points_layer)
            distance, closest_point_index = kdtree.query(self.mouse_position, k=1)

            self.update_point_layer_2(closest_point_index, 0, "remove")

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

    def create_point_layer(self, point, label):
        try:
            self.napari_viewer.layers.remove("Initial_Labels")
        except:
            pass

        self.napari_viewer.add_points(
            point,
            name="Initial_Labels",
            face_color="#00000000",
            properties={"label": label.astype(np.int16)},
            edge_color="label",
            edge_color_cycle=self.color_map_specified,
            edge_width=0.1,
            symbol="square",
            size=40,
        )
        self.napari_viewer.layers["Initial_Labels"].mode = "select"

    def update_point_layer_2(self, index, label, func):
        try:
            p_layer = self.napari_viewer.layers["Initial_Labels"]

            if func == "add":
                point_layer = p_layer.data
                labels = p_layer.properties["label"]

                points_layer = np.insert(point_layer, 0, self.mouse_position, axis=0)
                labels = np.insert(labels, 0, [label], axis=0)

                self.create_point_layer(points_layer, labels)
            elif func == "remove":
                point_layer = p_layer.data
                labels = p_layer.properties["label"]

                points_layer = np.delete(point_layer, index, axis=0)
                labels = np.delete(labels, index, axis=0)

                self.create_point_layer(points_layer, labels)
            elif func == "update":
                point_layer = p_layer.data
                labels = p_layer.properties["label"]
                if labels[index] != label:
                    labels[index] = label
                    self.create_point_layer(point_layer, labels)
            else:
                pass
            p_layer.edge_color_cycle = self.color_map_specified
            self._reset_view()
        except:
            pass

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
            frame = diff * (self.zoom_factor.value - 1)

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

    def _reset_view(self):
        self.napari_viewer.reset_view()
