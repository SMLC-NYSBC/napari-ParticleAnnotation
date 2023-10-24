import warnings
from os.path import splitext

import requests
from PyQt5.QtWidgets import QComboBox

from magicgui.widgets import (
    Container,
    VBox,
    create_widget,
    PushButton,
    LineEdit,
    Label,
    FloatSlider,
    Select,
    ComboBox,
)

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter

from napari import Viewer

import numpy as np
from qtpy.QtWidgets import QFileDialog

from napari.layers import Points
import napari

from ParticleAnnotation.cloud.aws_api import url, dir_

from ParticleAnnotation._qt.viewer_utils import (
    ViewerModel,
    QtViewerWrap,
    get_property_names,
    copy_layer,
    OwnPartial,
)
from ParticleAnnotation.cloud.utils import bytes_io_to_numpy_array
from ParticleAnnotation.utils.load_data import load_data_aws


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
        self.napari_viewer = napari_viewer

        self.file_list = []

        # Initialize model
        spacer1 = Label(value="-- Step 1: Initialize  Topaz  Active  learning --")
        self.load_ALM = ComboBox(name="Select Model", choices=())
        self.load_ALM.changed.connect(self._update_model_list)
        self.new_ALM = PushButton(name="New Model")
        self.new_ALM.clicked.connect(self._create_new_model)

        self.load_data = ComboBox(name="Load Data", choices=())
        self.load_data.changed.connect(self._load_file)
        self.send_data = PushButton(name="Send Data")
        self.send_data.clicked.connect(self._send_image_to_aws)

        self.init_data = PushButton(name="Initialize dataset")
        self.init_data.clicked.connect(self._initialize_model)

        spacer2 = Label(value="---------- Step 2: Iterative  Training ----------")
        self.num_particles_al = LineEdit(name="Num. of Particles", value="1")
        self.refresh = PushButton(name="Retrain")
        self.refresh.clicked.connect(self._refresh)
        self.reset_view = PushButton(name="Reset View")
        self.reset_view.clicked.connect(self._reset_view)

        spacer3 = Label(value="---------------- Step 3: Predict ----------------")
        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)
        self.update_model = PushButton(name="Update server model")
        self.update_model.clicked.connect(self._update_on_aws)

        spacer4 = Label(value="--------- Step 4: Visualize labels tool ---------")
        self.slide_pred = FloatSlider(
            name="Filter Particle",
            min=0,
            max=1,
        )
        self.slide_pred.changed.connect(self._filter_particle)

        # Space 1
        self.insert(1, spacer1)
        self.insert(
            2,
            VBox(
                widgets=(
                    self.load_ALM,
                    self.new_ALM,
                )
            ),
        )
        self.insert(
            3,
            VBox(
                widgets=(
                    self.load_data,
                    self.send_data,
                )
            ),
        )
        self.insert(4, VBox(widgets=(self.init_data,)))

        # Space 2
        self.insert(5, spacer2)
        self.insert(6, VBox(widgets=(self.num_particles_al, self.refresh)))
        self.insert(7, VBox(widgets=(self.reset_view,)))

        # Space 3
        self.insert(8, spacer3)
        self.insert(9, VBox(widgets=(self.predict, self.update_model)))

        # Space 4
        self.insert(10, spacer4)
        self.insert(11, VBox(widgets=(self.slide_pred,)))

        # Widget initialization
        self._update_data_list()

    def _load_file(self):
        response = requests.get(url + "getrawfiles", data={'f_name': self.load_data.value})
        image = bytes_io_to_numpy_array(response.json())

        load_data_aws(image)

    def _update_model_list(self):
        pass

    def _create_new_model(self):
        pass

    def _update_data_list(self):
        response = requests.get(url + "listfiles")

        if response.status_code == 200:
            self.file_list = response.json()
            file_list = [f[:5] for f in self.file_list]

            self.load_data.choices = tuple(file_list)
            self.load_data.value = file_list[0]

        else:
            print("Failed to fetch files:", response.status_code)
            print("Failed to fetch files:", response.text)

    def _send_image_to_aws(self):
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        root, extension = splitext(self.filename)

        format_ = extension[1:] if extension else None
        name_ = self.filename.split("/")[-1]

        files = {"file": (name_, open(f"{self.filename}", "rb"), f"image/{format_}")}

        response = requests.post(url + "uploadfile", files=files)

        if response.status_code == 200:
            print("File uploaded successfully:", response.json())
            self._update_data_list()
        else:
            print("Failed to upload file:", response.status_code, response.text)

    def _initialize_model(self):
        pass

    def _refresh(self):
        pass

    def _reset_view(self):
        pass

    def _predict(self):
        pass

    def _filter_particle(self):
        pass

    def _update_on_aws(self):
        pass
