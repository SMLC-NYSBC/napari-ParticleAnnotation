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
import numpy as np
from napari import Viewer
from napari.layers import Points
import napari
from scipy.spatial import KDTree

from napari.utils.notifications import show_info
from ParticleAnnotation.cloud.aws_api import url
from ParticleAnnotation.cloud.utils import bytes_io_to_numpy_array

class PredictionWidget(Container):
    def __init__(self, viewer_tm_score_3d: Viewer):
        """
        ToDo
            - centering of patches
            - marge initialize blr to re-train
            - make a clear gui
                - Load data, Train, Predict, Utils
            -
        """
        super().__init__(layout="vertical")
        self.napari_viewer = viewer_tm_score_3d

        # Global
        self.image_name = ""
        self.filename = None

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
        self.init = False
        self.AL_weights = None

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

        # ---------------- Initialize New Dataset -----------------
        self.box_size = LineEdit(name="Box", value="5")
        self.patch_size = LineEdit(name="Patch", value="128")
        self.pdb_id = LineEdit(name="PDB", value="7A4M")

        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)

        # ------------ Visualize labels tool & export -------------
        self.filter_particle_by_confidence = FloatSlider(
            name="Filter Particle",
            min=0,
            max=1,
        )
        self.filter_particle_by_confidence.changed.connect(
            self._filter_particle_by_confidence
        )

        # ---------------- Import & Export modules ----------------
        self.export_particles = PushButton(name="Save picks")
        self.export_particles.clicked.connect(self._export_particles)
        self.import_particles = PushButton(name="Load picks")
        self.import_particles.clicked.connect(self._import_particles)

        # ---------------- Load data ----------------
        self.load_data = ComboBox(name = "Load Data", choices = self._update_data_list())
        self.load_data_btt = PushButton(name = "Load Data")
        self.load_data_btt.clicked.connect(self._load_data)

        self.save_model = PushButton(name="Save Model")
        self.save_model.clicked.connect(self._save_model)
        self.load_model = PushButton(name="Load Model")
        self.load_model.clicked.connect(self._load_model)

        widget = VBox(
            widgets=(
                HBox(
                    widgets=(
                        self.pdb_id,
                        self.box_size,
                        self.patch_size,
                    )
                ),
                HBox(
                    widgets=(
                        self.save_model,
                        self.load_model,
                    )
                ),
                HBox(widgets=(self.predict,)),
                HBox(widgets=(self.filter_particle_by_confidence,)),
                HBox(
                    widgets=(
                        self.export_particles,
                        self.import_particles,
                    )
                ),
            )
        )

        self.napari_viewer.window.add_dock_widget(widget, area="left")

        self.device_ = get_device()
        show_info(f"Active learning model runs on: {self.device_}")

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
            name = self.napari_viewer.layers.selection.active.name

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
                show_info(
                    f"Warning: {e} error occurs while searching for {name} layer."
                )

    def key_event(self, viewer: Viewer, key: int):
        """
        Main key event definition.

        Args:
            key (int): Key definition of an action.
        """
        if self.activate_click:
            name = self.napari_viewer.layers.selection.active.name
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

