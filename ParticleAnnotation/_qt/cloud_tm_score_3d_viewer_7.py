import uuid
import requests
import numpy as np
from magicgui.widgets import (
    Container,
    PushButton,
    LineEdit,
    ComboBox,
    FloatSlider,
    Label,
    CheckBox,
    VBox,
    HBox,
)
from napari import Viewer
from napari.utils.notifications import show_info
from sklearn.neighbors import KDTree
from qtpy.QtWidgets import QFileDialog
from napari_bbox import BoundingBoxLayer

import torch
import pandas as pd

from ParticleAnnotation.utils.load_data import (
    load_coordinates,
)
from ParticleAnnotation.utils.scale import scale_image
from ParticleAnnotation.utils.model.utils import (
    correct_coord,
    get_device,
    get_random_patch,
)
from ParticleAnnotation.utils.viewer.viewer_functionality import (
    build_gird_with_particles,
)
import starfile

from ParticleAnnotation.cloud.utils import bytes_io_to_numpy_array
from ParticleAnnotation._qt.local_tm_score_3d_viewer import PlotPopup

colormap_for_display = "Spectral"

# # url = "http://localhost:8000/"  # Debugging
# url = "http://3.230.8.116:8000/"  # Production


class AWSWidget_7(Container):
    def __init__(self, viewer_cloud_tm_score_3d_7: Viewer):
        super(AWSWidget_7, self).__init__(layout="vertical")

        self.napari_viewer = viewer_cloud_tm_score_3d_7
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
        self.init, self.init_done, self.AL, self.Predict = True, False, False, False
        self.AL_weights = None
        self.weights_bias = None
        self.delta = None
        self.delta_values = [0.0]

        self.logits_full = None

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
        # "3.230.8.116"
        self.url = LineEdit(name="Server", value="3.230.8.116")
        self.resolution = CheckBox(name="High-Res", value=False)
        self.box_size = LineEdit(name="Model Mask Size [px]", value=5)
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

        spacer_2 = Label(value="------------------- Training -------------------")
        self.data_list = ComboBox(name="Dataset", choices=self._update_data_list())
        self.load_data = PushButton(name="Load")
        self.load_data.clicked.connect(self._select_particle_for_patches)
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

        self.import_particles = PushButton(name="Load particles")
        self.import_particles.clicked.connect(self._import_particles)

        self.save_model = PushButton(name="Save Model")
        self.save_model.clicked.connect(self._save_model)
        self.load_model = PushButton(name="Load Model")
        self.load_model.clicked.connect(self._load_model)

        spacer_6 = Label(value="-------------------- Restart -------------------")
        self.reset_session = PushButton(name="Restart Session")
        self.reset_session.clicked.connect(self._reset_session)

        widget = VBox(
            widgets=[
                HBox(
                    widgets=[
                        spacer_1,
                    ]
                ),
                HBox(
                    widgets=[
                        self.url,
                        self.resolution,
                    ]
                ),
                HBox(
                    widgets=[
                        self.box_size,
                        self.filter_size,
                    ]
                ),
                HBox(
                    widgets=[
                        self.pdb_id,
                        self.patch_size,
                    ]
                ),
                VBox(
                    widgets=[
                        self.pi,
                        self.gauss,
                    ]
                ),
                HBox(
                    widgets=[
                        spacer_2,
                    ]
                ),
                HBox(
                    widgets=[
                        self.data_list,
                        self.load_data,
                    ]
                ),
                VBox(
                    widgets=[
                        self.train_BLR_on_patch,
                    ]
                ),
                HBox(
                    widgets=[
                        spacer_4,
                    ]
                ),
                VBox(
                    widgets=[
                        self.predict,
                        self.filter_particle_by_confidence,
                    ],
                ),
                HBox(
                    widgets=[
                        spacer_3,
                    ]
                ),
                HBox(
                    widgets=[
                        self.show_tomogram,
                        self.show_active_learning_grid,
                        self.show_particle_grid,
                    ]
                ),
                HBox(
                    widgets=[
                        spacer_5,
                    ]
                ),
                VBox(
                    widgets=[
                        self.import_particles,
                    ]
                ),
                HBox(
                    widgets=[
                        self.export_particles_all,
                        self.export_particles_labeled,
                    ]
                ),
                HBox(
                    widgets=[
                        self.export_particles_predict,
                        self.export_particles_filter,
                    ]
                ),
                HBox(
                    widgets=[
                        self.save_model,
                        self.load_model,
                    ]
                ),
                VBox(
                    widgets=[
                        spacer_6,
                        self.reset_session,
                    ]
                ),
            ]
        )

        widget.max_width = 400
        self.napari_viewer.window.add_dock_widget(widget, area="right")

        self.device_ = get_device()
        show_info(f"Active learning model runs on: {self.device_}")
        self._reset_views()

    def _update_data_list(self):
        """
        API call to get a list of available tomogram files on the AWS EC2 instance

         Returns:
              tuple: List of available tomograms
        """
        # Call API for response
        try:
            response = requests.get(
                url=f"http://{self.url.value}:8000/" + "list_tomograms",
                params={"dataset": "7"},
            )

            # If server response is successful
            if response.status_code == 200:
                self.file_list = response.json()

                return tuple(self.file_list)
            else:
                show_info(
                    f"Connection Error to http://{self.url.value}:8000/. With error code: {response.status_code}."
                )
                return ()

        except:
            show_info(
                f"Connection Error to http://{self.url.value}:8000/. Check if server is running."
            )
            return ()

    """""" """""" """""" """
    Main triggers for GUI
    """ """""" """""" """"""

    def _select_particle_for_patches(self):
        """
        Starting function - for the plugin. Function is allowing user to label
        particles of interest, and store them.
        """
        self.image_name = self.filename = self.data_list.value

        if self.tm_list is None:
            try:
                response = requests.get(
                    f"http://{self.url.value}:8000/" + "list_templates",
                    params={"tomo_name": self.image_name, "dataset": "7"},
                    timeout=None,
                )

                if response.status_code == 200:
                    self.tm_list = response.json()
                else:
                    show_info(
                        f"Connection Error to http://{self.url.value}:8000/. With error code: {response.status_code}."
                    )
                    self.tm_list = None
            except:
                self.tm_list = None
                show_info(
                    f"Connection Error to http://{self.url.value}:8000/. Check if server is running."
                )

        if not self.init_done:
            self.init = True
            self.all_grid = False
            self.grid_labeling_mode = False

            # Call API for response
            response = requests.get(
                url=f"http://{self.url.value}:8000/get_raw_tomos",
                params={
                    "f_name": self.image_name,
                    "dataset": "7",
                    "high_res": int(self.resolution.value),
                },
                timeout=None,
            )

            # Decode bytes to np.array
            self.img_process = bytes_io_to_numpy_array(response.content).astype(
                np.uint8
            )

            if not self.resolution.value:
                self.img_process, _ = scale_image(
                    scale=[250, 1440, 1022],
                    image=self.img_process,
                    nn=False,
                    device="cpu",
                )

            self.create_image_layer(
                self.img_process, name=self.image_name, transparency=False
            )

            try:
                if self.pdb_id.value == "6R7M":
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
                        id_
                        for id_, i in enumerate(self.tm_list)
                        if i.split("_")[1] == self.pdb_id.value
                    ][0]
            except:
                pass

            response = requests.get(
                url=f"http://{self.url.value}:8000/get_raw_templates",
                params={
                    "f_name": self.image_name,
                    "dataset": "7",
                    "pdb_name": self.tm_list[self.tm_idx],
                    "high_res": int(self.resolution.value),
                },
                timeout=None,
            )

            # Decode bytes to np.array
            self.tm_scores = bytes_io_to_numpy_array(response.content)
            if not self.resolution.value:
                self.tm_scores, _ = scale_image(
                    scale=[250, 1440, 1022],
                    image=self.tm_scores,
                    nn=False,
                    device="cpu",
                )

            if self.tm_scores.ndim == 4:
                self.tm_scores = self.tm_scores[0, ...].astype(np.uint8)

            self.create_image_layer(
                self.tm_scores,
                name="TM_Scores",
                transparency=False,
                visibility=False,
            )

            self._pdb_id_update()
        else:
            return

    def _train_BLR_on_patch(self):
        if len(self.user_annotations) == 0:
            show_info("Please label any particle first!")
            return

        self.init_done = True
        self.init = False
        self.AL = True

        self._reset_views()
        self.all_grid = False
        self.grid_labeling_mode = False

        patch_size = int(self.patch_size.value)
        box_size = int(self.box_size.value)

        """If Patch do not exist create one and drawn panicles"""
        self.patch_corner = get_random_patch(
            self.img_process.shape, patch_size, self.user_annotations[:, :3]
        )

        if self.weights_bias is None:
            features = len(self.tm_list)
            if "score_ice" in self.tm_list:
                features += 2

            weights = np.zeros((features,), dtype=np.float32)
            bias = np.zeros((1,), dtype=np.float32)
        else:
            weights = self.weights_bias[0]
            bias = self.weights_bias[1]

        self.clean_viewer()
        response = requests.get(
            f"http://{self.url.value}:8000/" + "re_train_model",
            params={
                "f_name": self.image_name,
                "dataset": "7",
                "tm_idx": self.tm_idx,
                "patch_corner": ",".join(map(str, self.patch_corner)),
                "patch_size": patch_size,
                "box_size": box_size,
                "pi": float(self.pi.value),
                "weights": ",".join(map(str, weights)),
                "bias": ",".join(map(str, bias)),
                "points": ",".join(map(str, self.user_annotations.flatten())),
            },
            timeout=None,
        )
        self.weights_bias = response.json().split("|")
        self.weights_bias = [
            [float(i) for i in self.weights_bias[0].split(",")],
            [float(i) for i in self.weights_bias[1].split(",")],
        ]

        w = np.array(self.weights_bias[0])
        if self.delta is None:
            show_info("Training weights delta = 0.0")
        else:
            self.delta_values.append(np.abs(np.mean(self.delta - w)))

            print(f"Training weights delta = {self.delta_values[-1]}")

        self.delta_plot.update_plot(y_values=self.delta_values)
        self.delta = w

        """Draw new patch and find new particle for user to label"""
        response = requests.get(
            f"http://{self.url.value}:8000/" + "new_proposal",
            params={
                "f_name": self.image_name,
                "dataset": "7",
                "patch_corner": ",".join(map(str, self.patch_corner)),
                "patch_size": patch_size,
                "pi": float(self.pi.value),
                "weights": ",".join(map(str, self.weights_bias[0])),
                "bias": ",".join(map(str, self.weights_bias[1])),
            },
            timeout=None,
        )
        response = response.json().split("|")

        logits_shape = tuple([int(i) for i in response[1].split(",")])
        self.logits_patch = np.array([int(i) for i in response[0].split(",")])
        self.logits_patch = self.logits_patch.reshape(logits_shape)

        self.patch_points = np.array([float(i) for i in response[2].split(",")])
        self.patch_points = self.patch_points.reshape(len(self.patch_points) // 3, 3)

        self.patch_label = np.zeros((self.patch_points.shape[0],))
        self.patch_label[:,] = 2

        self._show_active_learning_grid()

    def _predict(self):
        if self.weights_bias is None:
            show_info(
                "You must load model or pre-train one with Active-Learning protocols!"
            )

        self.AL = False
        self.Predict = True

        self.all_grid = False
        self.grid_labeling_mode = False
        self.clean_viewer()
        self._reset_views()

        patch_size = int(self.patch_size.value)
        filter_size = int(self.filter_size.value)
        gauss_filter = float(self.gauss.value)

        response = requests.get(
            f"http://{self.url.value}:8000/" + "predict",
            params={
                "f_name": self.image_name,
                "dataset": "7",
                "high_res": int(self.resolution.value),
                "pbd_id": self.pdb_id.value,
                "patch_size": patch_size,
                "pi": float(self.pi.value),
                "weights": ",".join(map(str, self.weights_bias[0])),
                "bias": ",".join(map(str, self.weights_bias[1])),
                "gauss_filter": gauss_filter,
                "filter_size": filter_size,
            },
            timeout=None,
        )
        response = response.json().split("|")

        shape_ = tuple([int(i) for i in response[1].split(",")])
        self.logits_full = np.array([int(i) for i in response[0].split(",")])
        self.logits_full = self.logits_full.reshape(shape_)

        if not self.resolution.value:
            self.logits_full, _ = scale_image(
                scale=[250, 1440, 1022], image=self.logits_full, nn=False, device="cpu"
            )

        peaks = np.array([float(i) for i in response[2].split(",")])

        self.peaks_full = peaks.reshape((len(peaks) // 3, 3))
        self.peaks_confidence_full = np.array(
            [float(i) for i in response[3].split(",")]
        )

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

        self.weights_bias == None

        # Particles selections
        self.cur_proposal_index, self.proposals = 0, []
        self.user_annotations = np.zeros((0, 4))  # Z, Y, X, Label
        self.selected_particle_id = None

        self.delta_values = [0.0]
        self.delta_plot.update_plot(y_values=self.delta_values)

        # Remove after testing
        self.particle = None
        self.confidence = None
        self.patch_points, self.patch_label = np.zeros((0, 3)), np.zeros((1,))

        # BLR model
        self.model, self.model_pred, self.weights, self.bias = None, None, None, None
        self.init, self.init_done, self.AL, self.Predict = True, False, False, False
        self.AL_weights = None

    """""" """""" """""" """
    Viewer functionality
    """ """""" """""" """"""

    def _pdb_id_update(self):
        self._soft_reset_session()

        try:
            if self.pdb_id.value == "6R7M":
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
                        if i.endswith(self.pdb_id.value)
                    ][0]
            else:
                self.tm_idx = [
                    id_
                    for id_, i in enumerate(self.tm_list)
                    if i.endswith(self.pdb_id.value)
                ][0]
        except:
            return

        if self.init:
            filter_size = int(self.filter_size.value)
            # Restart user annotation storage
            self.user_annotations = np.zeros((0, 4))

            response = requests.get(
                f"http://{self.url.value}:8000/" + "get_initial_peaks",
                params={
                    "f_name": self.image_name,
                    "dataset": "7",
                    "filter_size": filter_size,
                    "tm_idx": self.tm_idx,
                },
            )

            if response.status_code == 200:
                peaks = bytes_io_to_numpy_array(response.content)
            else:
                peaks = np.zeros((1, 4))

            # self.user_annotations = peaks
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

        # ToDo move to the Cloud
        if self.logits_full is None:
            if self.weights_bias is not None:
                response = requests.get(
                    f"http://{self.url.value}:8000/" + "show_tomogram",
                    params={
                        "f_name": self.image_name,
                        "dataset": "7",
                        "high_res": int(self.resolution.value),
                        "patch_corner": ",".join(map(str, self.patch_corner)),
                        "patch_size": patch_size,
                        "pi": float(self.pi.value),
                        "weights": ",".join(map(str, self.weights_bias[0])),
                        "bias": ",".join(map(str, self.weights_bias[1])),
                        "gauss_filter": gauss_filter,
                        "filter_size": filter_size,
                    },
                    timeout=None,
                )
                response = response.json().split("|")
                shape_ = tuple([int(i) for i in response[1].split(",")])
                self.logits_full = np.array([int(i) for i in response[0].split(",")])
                self.logits_full = self.logits_full.reshape(shape_)

                if not self.resolution.value:
                    self.logits_full, _ = scale_image(
                        scale=[250, 1440, 1022],
                        image=self.logits_full,
                        nn=False,
                        device="cpu",
                    )

                peaks = np.array([float(i) for i in response[2].split(",")])
                peaks = peaks.reshape((len(peaks) // 3, 3))

                peaks_confidence = np.array([float(i) for i in response[3].split(",")])

                self.peaks_full = peaks[-100:, :]
                self.peaks_confidence_full = peaks_confidence[-100:]

        self.create_image_layer(self.img_process, name="Tomogram_Patch")

        self.create_image_layer(
            self.tm_scores,
            name="TM_Scores",
            transparency=True,
            visibility=False,
        )

        if self.logits_full is not None:
            self.create_image_layer(
                self.logits_full, name="Sigmoid", transparency=True, visibility=False
            )

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
                    self.tm_scores[None, ...]
                    if self.logits_patch is None
                    else self.logits_patch[None, ...]
                ),
                0,
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
                    self.tm_scores[None, ...]
                    if self.logits_full is None
                    else self.logits_full[None, ...]
                ),
                0,
                int(self.box_size.value),
                correct=False,
            )
        else:
            if len(self.user_annotations) > 0:
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
                        self.tm_scores[None, ...]
                        if self.logits_full is None
                        else self.logits_full[None, ...]
                    ),
                    0,
                    int(self.box_size.value),
                    correct=False,
                )
            else:
                return

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

    def _export(self, data: np.ndarray, name: str):
        """
        General export function to .star file format
        """
        # Save only user annotations (positive labels)
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File",
            directory=name,
        )

        data = pd.DataFrame(
            data=data,
            columns=[
                "_rlnCoordinateZ",
                "_rlnCoordinateY",
                "_rlnCoordinateX",
                "_rlnConfidence",
                "_rlnTomogramName",
                "_rlnGroup",
                "_rlnUserID",
                "_rlnPDBID",
            ],
        )
        starfile.write(data, filename)

    def _export_particles_all(self):
        user_annotations = self.user_annotations.copy()
        groups = np.repeat("user_annotations", len(user_annotations))

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

            groups = np.hstack((groups, np.repeat("predictions", len(prediction))))

        data = np.hstack((data, np.repeat(self.image_name, len(data))[:, None]))
        data = np.hstack((data, groups[:, None]))
        data = np.hstack(
            (data, np.repeat(str(uuid.UUID(int=uuid.getnode())), len(data))[:, None])
        )
        data = np.hstack((data, np.repeat(self.pdb_id.value, len(data))[:, None]))

        print(data)
        self._export(data, "all_annotations.star")

    def _export_particles_labeled(self):
        data = self.user_annotations.copy()

        data = np.hstack((data, np.repeat(self.image_name, len(data))[:, None]))
        data = np.hstack((data, np.repeat("user_annotations", len(data))[:, None]))
        data = np.hstack(
            (data, np.repeat(str(uuid.UUID(int=uuid.getnode())), len(data))[:, None])
        )
        data = np.hstack((data, np.repeat(self.pdb_id.value, len(data))[:, None]))

        self._export(data, "user_annotations.star")

    def _export_particles_predict(self):
        try:
            prediction_particle = self.napari_viewer.layers["Particle_Prediction"].data
            prediction_labels = self.napari_viewer.layers[
                "Particle_Prediction"
            ].properties["label"]

            prediction = np.hstack((prediction_particle, prediction_labels[:, None]))

            prediction = np.hstack(
                (prediction, np.repeat(self.image_name, len(prediction))[:, None])
            )
            prediction = np.hstack(
                (prediction, np.repeat("user_annotations", len(prediction))[:, None])
            )
            prediction = np.hstack(
                (
                    prediction,
                    np.repeat(str(uuid.UUID(int=uuid.getnode())), len(prediction))[
                        :, None
                    ],
                )
            )
            prediction = np.hstack(
                (prediction, np.repeat(self.pdb_id.value, len(prediction))[:, None])
            )

            self._export(prediction, f"{self.pdb_id.value}_prediction.star")
        except:
            pass

    def _export_particles_filter(self):
        try:
            prediction_particle = self.napari_viewer.layers[
                "Particle_Prediction_Filtered"
            ].data
            prediction_labels = self.napari_viewer.layers[
                "Particle_Prediction_Filtered"
            ].properties["label"]

            prediction = np.hstack((prediction_particle, prediction_labels[:, None]))

            prediction = np.hstack(
                (prediction, np.repeat(self.image_name, len(prediction))[:, None])
            )
            prediction = np.hstack(
                (prediction, np.repeat("user_annotations", len(prediction))[:, None])
            )
            prediction = np.hstack(
                (
                    prediction,
                    np.repeat(str(uuid.UUID(int=uuid.getnode())), len(prediction))[
                        :, None
                    ],
                )
            )
            prediction = np.hstack(
                (prediction, np.repeat(self.pdb_id.value, len(prediction))[:, None])
            )

            self._export(prediction, f"{self.pdb_id.value}_prediction_filtered.star")

        except:
            pass

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
            # Check for AL_weights
            if self.weights_bias is not None:
                filename, _ = QFileDialog.getSaveFileName(
                    caption="Save File", directory="Active_learn_model.pt"
                )

                self.weights_bias = [
                    torch.Tensor(self.weights_bias[0]),
                    torch.Tensor(self.weights_bias[1]),
                ]

            torch.save(self.weights_bias, filename)

    def _load_model(self):
        """
        Function to load and update self.AL_weights which is a list [self.weight, self.bias]
        expected as a pickle torch .pt file.

        If self.model is not None, update model weights. Else create self.model with
        this weights.
        """
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        self.AL_weights = torch.load(f"{self.filename}")

        if self.weights_bias is None:
            self.weights_bias = self.AL_weights

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
                name = self.napari_viewer.layers.selection.active.name
                point_layer = self.napari_viewer.layers[name]

                self.user_annotations = np.delete(self.user_annotations, index, axis=0)

                points = point_layer.data
                labels = point_layer.properties["label"]

                points = np.delete(points, index, axis=0)
                labels = np.delete(labels, index, axis=0)
            else:
                self.patch_points = np.delete(self.patch_points, index, axis=0)
                self.patch_label = np.delete(self.patch_label, index, axis=0)

                points = np.delete(points, index, axis=0)
                labels = np.delete(labels, index, axis=0)

                idx = point in self.user_annotations[:, :3]
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
