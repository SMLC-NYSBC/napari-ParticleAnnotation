from os.path import splitext

import requests

from magicgui.widgets import (
    Container,
    VBox,
    PushButton,
    LineEdit,
    Label,
    FloatSlider,
    ComboBox,
    HBox,
)
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter

from napari import Viewer

import numpy as np
from qtpy.QtWidgets import QFileDialog

from napari.layers import Points
import napari
from scipy.spatial import KDTree

from ParticleAnnotation.cloud.aws_api import url

from ParticleAnnotation._qt.viewer_utils import (
    ViewerModel,
    QtViewerWrap,
    get_property_names,
    copy_layer,
    OwnPartial,
)
from ParticleAnnotation.cloud.utils import bytes_io_to_numpy_array


class MultipleViewerWidget(QSplitter):
    def __init__(self, viewer_cloud: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer_cloud
        self.viewer_model1 = ViewerModel(title="View_1")
        self.viewer_model2 = ViewerModel(title="View_2")
        self._block = False
        self.qt_viewer1 = QtViewerWrap(self.viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(self.viewer, self.viewer_model2)

        self.points_layer = self.viewer_model1.add_points(
            [0, 0, 0], name="Mouse Pointer", symbol="cross", size=2
        )
        self.annotation_widget = AnnotationWidgetv2(self.viewer)
        self.viewer.window.add_dock_widget(
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

        """Global variables"""
        self.file_list = []
        self.model_name, self.model_type = None, None
        self.color_map_specified = {0.0: "#D81B60", 1.0: "#1E88E5", 2.0: "#FFC107"}
        self.activate_click = False

        """"Widget elements"""
        spacer1 = Label(value="-- Step 1: Initialize  Topaz  Active  learning --")
        self.load_ALM = ComboBox(name="Select Model", choices=self._update_model_list())
        self.load_ALM_btt = PushButton(name="Load")
        self.load_ALM_btt.changed.connect(self._load_model)
        self.new_ALM = PushButton(name="New Model")
        self.new_ALM.clicked.connect(self._create_new_model)

        self.load_data = ComboBox(name="Load Data", choices=self._update_data_list())
        self.load_data_btt = PushButton(name="Load")
        self.load_data_btt.clicked.connect(self._load_file)
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
        self.update_model.clicked.connect(self._update_model_on_aws)

        spacer4 = Label(value="--------- Step 4: Visualize labels tool ---------")
        self.slide_pred = FloatSlider(
            name="Filter Particle",
            min=0,
            max=1,
        )
        self.slide_pred.changed.connect(self._filter_particle)

        """Widget layer orders"""
        # Space 1
        self.insert(1, spacer1)
        self.insert(
            2,
            VBox(
                widgets=(
                    HBox(
                        widgets=(
                            self.load_ALM,
                            self.load_ALM_btt,
                        )
                    ),
                    self.new_ALM,
                    HBox(
                        widgets=(
                            self.load_data,
                            self.load_data_btt,
                        )
                    ),
                    self.send_data,
                    self.init_data,
                )
            ),
        )

        # Space 2
        self.insert(3, spacer2)
        self.insert(
            4,
            VBox(
                widgets=(
                    HBox(widgets=(self.num_particles_al, self.refresh)),
                    self.reset_view,
                )
            ),
        )

        # Space 3
        self.insert(5, spacer3)
        self.insert(6, VBox(widgets=(self.predict, self.update_model)))

        # Space 4
        self.insert(7, spacer4)
        self.insert(8, VBox(widgets=(self.slide_pred,)))

        """Widget initialization"""
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
        self.click_add_point_callback = None

    def _update_model_list(self) -> tuple:
        """
        API call to get a list of available [.pth] files on the AWS EC2 instance

         Returns:
              tuple: List of available pre-trained models
        """
        # Call API for response
        try:
            response = requests.get(url + "list_models")

            # If server responded process the response
            if response.status_code == 200:
                self.model_list = response.json()

                return tuple(self.model_list)
            else:
                return ()
        except:
            show_info(f"Connection Error to {url}. Check if server is running.")
            return ()

    def _load_model(self):
        """

        Returns:
        """
        self.model_name = self.load_ALM.value

    def _create_new_model(self):
        """

        Return:
            str: Filename of the newly created model. This is needed for model update.
        """
        # Call API for response
        try:
            response = requests.get(url + "new_model")

            # Update model name and indicate we are working on a new model not pre-trained
            self.model_name = response.json()
            self.model_type = "New"
        except:
            show_info(f"Connection Error to {url}. Check if server is running.")
            return ()

    def _update_data_list(self) -> tuple:
        """
        API call to get a list of available image files on the AWS EC2 instance

        Note:
            Sent to local machine images are already down-sample to reduce the size.
            Shown image is just a visual, all calculations are handle by server.

         Returns:
              tuple: List of available pre-trained models
        """
        # Call API for response
        try:
            response = requests.get(url + "list_files")

            # If server responded process the response
            if response.status_code == 200:
                self.file_list = response.json()

                return tuple(self.file_list)
            else:
                return ()
        except:
            show_info(f"Connection Error to {url}. Check if server is running.")
            return ()

    def _load_file(self):
        """
        Load a down-sample image file to work on.

        Note:
            If we cannot establish the connection. Return random 2D file.

        Return:
             napari.viewer.add_image: New Image layer.
        """
        # Call API for response
        try:
            response = requests.get(
                url + "get_raw_files",
                params={"f_name": self.load_data.value},
                timeout=None,
            )

            # Decode bytes to np.array
            image = bytes_io_to_numpy_array(response.content)

            # Build new image layer
            self.napari_viewer.add_image(data=image, name="Raw_Image")
            self.napari_viewer.layers["Raw_Image"].contrast_limits = (
                image.min(),
                image.max(),
            )
            self._reset_view()
            print(f"Try Loaded: {self.load_data.value}")
        except:
            show_info(f"Connection Error to {url}. Check if server is running.")
            self.napari_viewer.add_image(
                data=np.random.random(size=(512, 512)), name="Connection_Error"
            )

    def _send_image_to_aws(self):
        """
        Send image file to AWS EC2 instance.

        Return:
            napari.show_info: User info prompt to indicate output.
        """
        # Package image file in correct format
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        root, extension = splitext(self.filename)

        format_ = extension[1:] if extension else None
        name_ = self.filename.split("/")[-1]

        files = {"file": (name_, open(f"{self.filename}", "rb"), f"image/{format_}")}

        # Call API for response
        try:
            response = requests.post(url + "upload_file", files=files)

            if response.status_code == 200:
                show_info(f"File uploaded successfully: {response.json()}")
            else:
                show_info(
                    "Failed to upload file: \n"
                    f"Error: {response.status_code} \n"
                    f"Message:{response.text}"
                )
        except:
            show_info(f"Connection Error to {url}. Check if server is running.")

    def _initialize_model(self):
        params = {
            "m_name": self.model_name,
            "f_name": self.load_data.value,
            "n_part": self.num_particles_al.value,
        }
        print(params)
        try:
            response = requests.post(url + "initialize_model", params=params)

            if response.status_code == 200:
                p_label = response.json()
                self.create_point_layer(p_label[:, 1:], p_label[:, 0])
                self.activate_click = True

                show_info(f"Model successfully initialized: {response.json()}")
            else:
                show_info(
                    "Failed to initialized model: \n"
                    f"Error: {response.status_code} \n"
                    f"Message:{response.text}"
                )
        except:
            show_info(f"Connection Error to {url}. Check if server is running.")
        pass

    def _refresh(self):
        pass

    def _reset_view(self):
        self.napari_viewer.reset_view()

    def _predict(self):
        pass

    def _filter_particle(self):
        pass

    def _update_model_on_aws(self):
        pass

    def track_mouse_position(self, viewer, event):
        self.mouse_position = event.position

    def move_selected_point(self, viewer, event):
        if self.activate_click:
            try:
                # if self.activate_click:
                points_layer = viewer.layers["Particles_Labels"].data

                # Calculate the distance between the mouse position and all points
                distances = np.linalg.norm(points_layer - self.mouse_position, axis=1)
                closest_point_index = distances.argmin()

                # Clear the current selection and Select the closest point
                if self.selected_particle_id != closest_point_index:
                    self.selected_particle_id = closest_point_index

                    viewer.layers["Particles_Labels"].selected_data = set()
                    viewer.layers["Particles_Labels"].selected_data.add(
                        closest_point_index
                    )
            except:
                pass

    def ZEvent(self, viewer):
        if self.activate_click:
            # if self.activate_click:
            points_layer = viewer.layers["Particles_Labels"].data

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
            points_layer = viewer.layers["Particles_Labels"].data

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
            points_layer = viewer.layers["Particles_Labels"].data

            # Calculate the distance between the mouse position and all points
            kdtree = KDTree(points_layer)
            distance, closest_point_index = kdtree.query(self.mouse_position, k=1)

            self.update_point_layer_2(closest_point_index, 0, "remove")

    def update_point_layer_2(self, index, label, func):
        try:
            p_layer = self.napari_viewer.layers["Particles_Labels"]

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

    def create_point_layer(self, point, labels):
        try:
            self.napari_viewer.layers.remove("Particles_Labels")
        except:
            pass

        self.napari_viewer.add_points(
            point,
            name="Particles_Labels",
            face_color="#00000000",
            properties={"label": labels.astype(np.int16)},
            edge_color="label",
            edge_color_cycle=self.color_map_specified,
            edge_width=0.1,
            symbol="square",
            size=40,
        )
        self.napari_viewer.layers["Particles_Labels"].mode = "select"
