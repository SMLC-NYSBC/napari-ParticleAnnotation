from enum import Enum
from functools import partial
import warnings

import torch
from magicgui.widgets import (
    Container,
    HBox,
    SpinBox,
    create_widget,
    ComboBox,
    PushButton,
    LineEdit,
    Label,
)
from napari import Viewer
from vispy.geometry import Rect

from napari.components import ViewerModel
from copy import deepcopy

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter

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
from napari_ParticleAnnotation.utils.model.utils import (
    rank_candidate_locations,
)

"""
TODO: AL interaction
TODO: Run model
TODO: Find pick maxima
"""

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
    # If napari version >= 0.4.16, use the optimized version of the function
    if NAPARI_GE_4_16:
        return copy_layer_le_4_16(layer, name)

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


class AnnotationWidget(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(layout="vertical")
        self.color_map_specified = {0: "red", 1: "green", 2: "black"}
        self.napari_viewer = napari_viewer
        self.click_add_point_callback = None

        # Control Downsampling factor
        options = [1, 2, 4, 8, 16]
        self.sampling_layer = ComboBox(
            name="Downsample_factor", value=options[3], choices=options
        )
        spacer1 = Label(value="------- Initialize Active learning model ------")
        spacer2 = Label(value="----------- Iteratively train model -----------")
        spacer3 = Label(value="------------ Visualize labels tool ------------")

        # Initialize model
        self.l2 = LineEdit(name="L2", value="1.0")
        self.pi = LineEdit(name="pi", value="0.01")
        self.pi_weight = LineEdit(name="pi_weight", value="1000")

        self.init_mrc = PushButton(name="Initialize Active Learning model")
        self.init_mrc.clicked.connect(self._init_mrc)

        # AL Buttons
        self.init_model = PushButton(name="Init. model")
        self.init_model.clicked.connect(self._init_model)
        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)
        self.prev = PushButton(name="Preview")
        self.prev.clicked.connect(self._preview)
        self.refresh = PushButton(name="Retrain")
        self.refresh.clicked.connect(self._refresh)
        self.next = PushButton(name="Next")
        self.next.clicked.connect(self._next)

        self.napari_viewer.bind_key("z", self._positive)
        self.napari_viewer.bind_key("x", self._negative)

        # Control particle viewing
        self.points_layer = create_widget(annotation=Points, label="ROI", options={})
        self.points_layer.changed.connect(self._update_roi_info)

        self.component_selector = SpinBox(name="Particle ID", min=0)
        self.component_selector.changed.connect(self._component_num_changed)

        self.zoom_factor = create_widget(
            annotation=float, label="Zoom factor", value=100
        )
        self.zoom_factor.changed.connect(self._component_num_changed)

        # Reset view
        self.reset_view = PushButton(name="Reset View")
        self.reset_view.clicked.connect(self._reset_view)

        layout0 = HBox(widgets=(self.sampling_layer,))
        layer_model = HBox(
            widgets=(
                self.l2,
                self.pi,
                self.pi_weight,
            )
        )
        layer_init = HBox(widgets=(self.init_mrc,))
        layer_model1 = HBox(widgets=(self.init_model, self.predict))
        layer_train = HBox(
            widgets=(
                self.prev,
                self.refresh,
                self.next,
            )
        )
        layout1 = HBox(
            widgets=(
                self.points_layer,
                self.component_selector,
            )
        )
        layout2 = HBox(
            widgets=(
                self.zoom_factor,
                self.reset_view,
            )
        )
        self.insert(0, spacer1)
        self.insert(1, layout0)
        self.insert(2, layer_model)
        self.insert(3, layer_init)
        self.insert(4, spacer2)
        self.insert(5, layer_model1)
        self.insert(6, layer_train)
        self.insert(7, spacer3)
        self.insert(8, layout1)
        self.insert(9, layout2)

    def _predict(self):
        active_layer_name = self.napari_viewer.layers.selection.active.name
        points_layer = self.napari_viewer.layers[f"{active_layer_name}"]

        labels = points_layer.properties["label"]
        data = np.asarray(points_layer.data)
        data = np.array((data[:, 0], data[:, 1], np.array(labels).astype(np.int8))).T

        if data[-1, 2] == 2:
            data = data[:-1, :]

        self.y = label_points_to_mask(data, self.shape)
        self.count = (~torch.isnan(self.y)).float()

        model_pred = BinaryLogisticRegression(self.x.shape[1], pi_weight=1000)
        model_pred.fit(self.x, self.y.ravel(), weights=self.count.ravel())

        with torch.no_grad():
            logits = model_pred(self.x).reshape(*self.shape)
            p = torch.sigmoid(logits)
        logits = logits.numpy()

        max_filter = maximum_filter(logits, size=15)
        peaks = logits - max_filter
        peaks = np.where(peaks == 0)
        peaks = np.stack(peaks, axis=-1)
        peak_logits = p[peaks[:, 0], peaks[:, 1]]

        # order = np.argsort(-peak_logits.ravel())

        self.napari_viewer.add_points(
            peaks,
            name=f"{active_layer_name}_predict",
            properties={"confidence": peak_logits},
            edge_color="black",
            face_color="confidence",
            face_colormap="viridis",
            edge_width=0.1,
            symbol="disc",
            size=5,
        )

    def _preview(self):
        pass

    def _refresh(self):
        active_layer_name = self.napari_viewer.layers.selection.active.name
        points_layer = self.napari_viewer.layers[f"{active_layer_name}"]

        # Retrive points labels
        try:
            labels = points_layer.properties["label"]
            data = np.asarray(points_layer.data)
            data = np.array(
                (data[:, 0], data[:, 1], np.array(labels).astype(np.int8))
            ).T
        except:
            data = []

        self.y = label_points_to_mask(data, self.shape)
        self.count = (~torch.isnan(self.y)).float()
        self.model.fit(self.x, self.y.ravel(), weights=self.count.ravel())

        self.cur_proposal_index, self.proposals = rank_candidate_locations(
            self.model, self.x, self.shape, self.proposals
        )

        self.add_point(self.proposals[self.cur_proposal_index])
        self.component_selector.value = len(data) + 1

    def _init_model(self):
        self.disable_click_add_point(self.napari_viewer)

        self.cur_proposal_index, self.proposals = 0, []

        active_layer_name = self.napari_viewer.layers.selection.active.name
        points_layer = self.napari_viewer.layers[f"{active_layer_name}"]

        # Retrive points labels
        try:
            labels = points_layer.properties["label"]
            data = np.asarray(points_layer.data)
            data = np.array(
                (data[:, 0], data[:, 1], np.array(labels).astype(np.int8))
            ).T
        except:
            data = []

        # update y and count
        self.y = label_points_to_mask(data, self.shape)
        self.count = torch.where(
            ~torch.isnan(self.y), torch.ones_like(self.y), torch.zeros_like(self.y)
        )

        self.model = BinaryLogisticRegression(
            n_features=self.x.shape[1],
            l2=float(self.l2.value),
            pi=float(self.pi.value),
            pi_weight=float(self.pi_weight.value),
        )
        self.model.fit(self.x, self.y.ravel(), weights=self.count.ravel())

        self.cur_proposal_index, self.proposals = rank_candidate_locations(
            self.model, self.x, self.shape, self.proposals
        )
        self.activate_click = False
        self.add_point(self.proposals[self.cur_proposal_index])
        self.component_selector.value = len(data) + 1

    def _next(self):
        pass

    def _update_roi_info(self):
        if not self.activate_click:
            self._component_num_changed()

    def _component_num_changed(self):
        self._zoom()

    def _reset_view(self):
        self.napari_viewer.reset_view()

    def _init_mrc(self):
        self.activate_click = True
        self.proposals = []
        self.cur_proposal_index = 0

        # Retrieve active image
        active_layer_name = self.napari_viewer.layers.selection.active.name
        img = self.napari_viewer.layers[active_layer_name]

        # Run some operation on the image data
        img_process = downsample(img.data, factor=self.sampling_layer.value)
        self.shape = img_process.shape
        _min, _max = np.quantile(img_process.ravel(), [0.1, 0.9])
        img.data = (img_process - (_max + _min) / 2) / (_max - _min)

        self.napari_viewer.layers[active_layer_name].contrast_limits = (
            img.data.min(),
            img.data.max(),
        )
        self._reset_view()

        self.x, _ = initialize_model(img.data)

        self.activate_click = True
        self.setup_click_add_point(self.napari_viewer)

    def setup_click_add_point(self, viewer):
        active_layer_name = viewer.layers.selection.active.name

        def click_add_point(viewer, event):
            # Get the coordinates of the clicked point
            coord = event.position

            # Set the label id depending on which button was clicked
            if event.button == 1:
                # Left button clicked, set label id to 1
                label = 1
            elif event.button == 2:
                # Right button clicked, set label id to 0
                label = 0
            else:
                label = None

            if label is not None:
                try:
                    points_layer = self.napari_viewer.layers[
                        f"{active_layer_name}_label"
                    ]
                    if not isinstance(points_layer, Points):
                        raise ValueError("Expected a Points layer")

                    # Add the clicked point to the points layer and add label
                    points_layer.add(coord)
                    points_layer.properties["label"][-1] = label
                    coord = points_layer.data
                    label = points_layer.properties["label"]

                    viewer.layers.remove(f"{active_layer_name}_label")
                    self.napari_viewer.add_points(
                        coord,
                        name=f"{active_layer_name}_label",
                        face_color="#00000000",
                        properties={"label": np.array(label)},
                        edge_color="label",
                        edge_color_cycle=self.color_map_specified,
                        edge_width=0.1,
                        symbol="square",
                        size=40,
                    )
                except:
                    self.napari_viewer.add_points(
                        [coord],
                        name=f"{active_layer_name}_label",
                        face_color="#00000000",
                        properties={"label": [label]},
                        edge_color="label",
                        edge_color_cycle=self.color_map_specified,
                        edge_width=0.1,
                        symbol="square",
                        size=40,
                    )
                self._reset_view()

        # Store the callback in an instance variable
        self.click_add_point_callback = click_add_point

        # Add the callback to the viewer
        viewer.mouse_drag_callbacks.append(self.click_add_point_callback)

    def disable_click_add_point(self, viewer):
        # Check if the callback has been added before
        if self.click_add_point_callback in viewer.mouse_drag_callbacks:
            # Remove the callback from the viewer
            viewer.mouse_drag_callbacks.remove(self.click_add_point_callback)

    def add_point(self, coord):
        active_layer_name = self.napari_viewer.layers.selection.active.name

        try:
            points_layer = self.napari_viewer.layers[f"{active_layer_name}"]
            points_layer.add(coord)
            coord = points_layer.data
            points_layer.properties["label"][-1] = 2
            label = points_layer.properties["label"]

            self.napari_viewer.layers.remove(f"{active_layer_name}")
            self.napari_viewer.add_points(
                coord,
                name=f"{active_layer_name}",
                face_color="#00000000",
                properties={"label": np.array(label)},
                edge_color="label",
                edge_color_cycle=self.color_map_specified,
                edge_width=0.1,
                symbol="square",
                size=40,
            )
        except:
            self.napari_viewer.add_points(
                [coord],
                name=f"{active_layer_name}",
                face_color="#00000000",
                properties={"label": np.array([2])},
                edge_color="label",
                edge_color_cycle=self.color_map_specified,
                edge_width=0.1,
                symbol="square",
                size=40,
            )

    def update_last_label(self, label_id):
        active_layer_name = self.napari_viewer.layers.selection.active.name
        points_layer = self.napari_viewer.layers[f"{active_layer_name}"]

        coord = points_layer.data
        label = points_layer.properties["label"]
        points_layer.properties["label"][-1] = label_id

        self.napari_viewer.layers.remove(f"{active_layer_name}")
        self.napari_viewer.add_points(
            coord,
            name=f"{active_layer_name}",
            face_color="#00000000",
            properties={"label": np.array(label)},
            edge_color="label",
            edge_color_cycle=self.color_map_specified,
            edge_width=0.1,
            symbol="square",
            size=40,
        )

    def _positive(self, viewer):
        # Update label
        self.update_last_label(1)

        # Update model and draw new suggestion
        self._refresh()

    def _negative(self, viewer):
        # Update label
        # Retrain and draw new one
        self.update_last_label(0)

        # Update model and draw new suggestion
        self._refresh()

    def _update_point(self, lower_bound, upper_bound):
        point = (lower_bound + upper_bound) / 2
        current_point = self.napari_viewer.dims.point[-len(lower_bound) :]
        dims = len(lower_bound) - self.napari_viewer.dims.ndisplay
        start_dims = self.napari_viewer.dims.ndim - len(lower_bound)
        for i in range(dims):
            if not (lower_bound[i] <= current_point[i] <= upper_bound[i]):
                self.napari_viewer.dims.set_point(start_dims + i, point[i])

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

        self.annotation_widget = AnnotationWidget(viewer)
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

    def _reset_view(self):
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _sync_view(self):
        self.viewer_model1.camera.zoom = (
            self.viewer_model2.camera.zoom
        ) = self.viewer.camera.zoom

        slice_dim = self.viewer.camera.center

        self.viewer_model1.camera.center = (
            self.viewer.camera.center[0],
            self.viewer_model1.camera.center[1],
            self.viewer.camera.center[2],
        )
        self.viewer_model1.dims.set_point(1, slice_dim[1])

        self.viewer_model2.camera.center = (
            self.viewer.camera.center[0],
            self.viewer.camera.center[1],
            self.viewer_model2.camera.center[2],
        )
        self.viewer_model2.dims.set_point(-1, slice_dim[2])

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

        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = order
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
