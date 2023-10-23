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
)

from ParticleAnnotation.utils.load_data import downsample
from ParticleAnnotation.utils.model.utils import rank_candidate_locations
from ParticleAnnotation._qt.viewer_utils import (
    ViewerModel,
    QtViewerWrap,
    get_property_names,
    copy_layer,
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
