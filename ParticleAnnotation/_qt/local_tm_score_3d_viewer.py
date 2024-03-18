import numpy as np
from magicgui.widgets import (
    Container,
    PushButton,
    LineEdit,
    FloatSlider,
    VBox,
    HBox,
)
from napari import Viewer
from napari.utils.notifications import show_info
from sklearn.neighbors import KDTree

from ParticleAnnotation.utils.model.utils import get_device


class AnnotationWidget(Container):
    def __init__(self, viewer_tm_score_3d: Viewer):
        super().__init__(layout="vertical")
        self.napari_viewer = viewer_tm_score_3d

        # Global
        self.color_map_particle_classes = {
            0.0: "#D81B60",
            1.0: "#1E88E5",
            2.0: "#FFC107",
        }
        self.activate_user_clicks = False
        self.image_name = ""
        self.filename = None

        # Particles selections
        self.cur_proposal_index, self.proposals = 0, []
        self.chosen_particles = []
        self.curr_layer = "Chosen Particles"
        self.true_labels = np.array([])
        self.particle_list = []
        self.selected_particle_id = None

        # BLR model
        self.model, self.model_pred, self.weights, self.bias = None, None, None, None
        self.init = False
        self.AL_weights = None

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

        # ------- Step 1: Initialize New Dataset ------
        self.image_resolution = LineEdit(name="Pixel", value="8.0")
        self.box_size = LineEdit(name="Box", value="5")
        self.patch_size = LineEdit(name="Patch", value="128")

        self.select_particle_for_patches = PushButton(name="Select particles")
        self.select_particle_for_patches.clicked.connect(
            self._select_particle_for_patches
        )
        self.initialize_BLR = PushButton(name="Start Training")
        self.initialize_BLR.clicked.connect(self._initialize_BLR)

        self.save_model = PushButton(name="Save Model")
        self.save_model.clicked.connect(self._save_model)
        self.load_model = PushButton(name="Load Model")
        self.load_model.clicked.connect(self._load_model)

        # ------ Step 2: Initialize Active learning model -------
        self.train_BLR_on_patch = PushButton(name="Change Patch")
        self.train_BLR_on_patch.clicked.connect(self._train_BLR_on_patch)
        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)

        # ------------ Step 3: Visualize labels tool & export ------------
        self.filter_particle_by_confidence = FloatSlider(
            name="Filter Particle",
            min=0,
            max=1,
        )
        self.filter_particle_by_confidence.changed.connect(
            self._filter_particle_by_confidence
        )

        self.export_particles = PushButton(name="Export picks")
        self.export_particles.clicked.connect(self._export_particles)
        self.import_particles = PushButton(name="Import picks")
        self.import_particles.clicked.connect(self._import_particles)

        widget = VBox(
            widgets=(
                HBox(
                    widgets=(
                        self.image_resolution,
                        self.box_size,
                        self.patch_size,
                    )
                ),
                HBox(
                    widgets=(
                        self.select_particle_for_patches,
                        self.initialize_BLR,
                    )
                ),
                HBox(
                    widgets=(
                        self.save_model,
                        self.load_model,
                    )
                ),
                HBox(
                    widgets=(
                        self.train_BLR_on_patch,
                        self.predict,
                    )
                ),
                HBox(widgets=(self.filter_particle_by_confidence,)),
                HBox(widgets=(self.export_particles, self.import_particles)),
            )
        )

        self.napari_viewer.window.add_dock_widget(widget, area="left")

        device_ = get_device()
        show_info(f"Active learning model runs on: {device_}")

    """
    Mouse and keys bindings
    """

    def track_mouse_position(self, viewer, event):
        """
        Mouse binding helper function to update stored mouse position when it moves
        """
        self.mouse_position = event.position

    def selected_point_near_mouse(self, viewer, event):
        """
        Mouse binding helper function to select point near the mouse pointer
        """
        if self.activate_click:
            name = self.napari_viewer.layers.selection.active.name

            try:
                # if self.activate_click:
                points_layer = viewer.layers[name].data

                # TODO Filter points_layer and search only for points withing radius
                # TODO Just in case we have thousands or millions of points issue

                # Calculate the distance between the mouse position and all points
                distances = np.linalg.norm(points_layer - self.mouse_position, axis=1)
                closest_point_index = distances.argmin()

                # Clear the current selection and Select the closest point
                if self.selected_particle_id != closest_point_index:
                    self.selected_particle_id = closest_point_index

                    viewer.layers[name].selected_data = set()
                    viewer.layers[name].selected_data.add(
                        closest_point_index
                    )
            except Exception as e:
                print(f"Warning: {e} error occurs while searching for {name} layer.")

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

                if key == 2:
                    if distance[0] < 10:
                        self.update_point_layer(closest_point[0], 0, "remove")
                else:
                    if distance[0] > 10:
                        self.update_point_layer(None, key, "add")
                    else:
                        self.update_point_layer(closest_point[0], key, "update")

    def ZEvent(self, viewer):
        self.key_event(viewer, 0)

    def XEvent(self, viewer):
        self.key_event(viewer, 1)

    def CEvent(self, viewer):
        self.key_event(viewer, 2)

    """
    Main triggers for GUI
    """

    def _select_particle_for_patches(
        self,
    ):
        self.image_name = (
            self.filename
        ) = self.napari_viewer.layers.selection.active.name

        self.create_point_layer(
            np.array([]), np.array([]), name="Chosen_Particles_of_Interest"
        )
        self.activate_click = True

    def _initialize_BLR(
        self,
    ):
        pass

    def _train_BLR_on_patch(
        self,
    ):
        pass

    def _predict(
        self,
    ):
        pass

    """
    BLR Model helper functions
    """

    def _save_model(
        self,
    ):
        """
        Function to fetch self.AL_weights which is a list [self.weight, self.bias]
        and save it as a pickle torch .pt file
        """
        # TODO Navya
        pass

    def _load_model(
        self,
    ):
        """
        Function to load and update self.AL_weights which is a list [self.weight, self.bias]
        expected as a pickle torch .pt file.

        If self.model is not None, update model weights. Else create self.model with
        this weights.
        """
        # TODO Navya
        pass

    """
    Viewer functionality
    """

    """
    Viewer helper functionality
    """

    def create_point_layer(
        self, point: np.ndarray, label: np.ndarray, name="Initial_Labels"
    ):
        """
        Create a point layer in napari with 2D/3D points and associated labels.
        """

        # If layer of the same exist remove it for update
        # Overwriting layer results in not correctly displayed points labels
        try:
            self.napari_viewer.layers.remove(name)
        except Exception as e:
            print(f"Warning: {e} error occurs while searching for {name} layer.")

        if point.shape[0] > 0:
            self.napari_viewer.add_points(
                point,
                name=name,
                face_color="#00000000",  # Hex + alpha
                properties={"label": label.astype(np.int16)},
                edge_color="label",
                edge_color_cycle=self.color_map_particle_classes,
                edge_width=0.1,
                symbol="square",
                size=40,
            )
            self.napari_viewer.layers[name].mode = "select"
        else:  # Create empty layer
            self.napari_viewer.add_points(
                point,
                name=name,
                face_color="#00000000",  # Hex + alpha
                edge_color_cycle=self.color_map_particle_classes,
                edge_width=0.1,
                symbol="square",
                size=40,
            )
            self.napari_viewer.layers[name].mode = "select"

    def update_point_layer(self, index=None, label=0, func="add"):
        name = self.napari_viewer.layers.selection.active.name
        point_layer = self.napari_viewer.layers[name]

        # Add point pointed by mouse
        if func == "add":
            points = point_layer.data

            if points.shape[0] == 0:
                labels = np.array([label])
                points = np.array([self.mouse_position])
            else:
                labels = point_layer.properties["label"]
                labels = np.insert(labels, len(points), [label], axis=0)

                points = np.insert(points, len(points), self.mouse_position, axis=0)

            self.create_point_layer(points, labels, name)
        elif func == "remove":  # Remove point pointed by mouse
            points = point_layer.data
            labels = point_layer.properties["label"]

            points = np.delete(points, index, axis=0)
            labels = np.delete(labels, index, axis=0)

            self.create_point_layer(points, labels, name)
        elif func == "update":  # Update point pointed by mouse
            points = point_layer.data
            labels = point_layer.properties["label"]
            if labels[index] != label:
                labels[index] = label
                self.create_point_layer(points, labels, name)

        point_layer.edge_color_cycle = self.color_map_particle_classes

    """
    Global helper functions
    """

    def _filter_particle_by_confidence(
        self,
    ):
        """
        Function to fetch
        self.napari_viewer.layers.selection.active.name["Prediction_Filtered"]
        and filter particle based on the confidence scored given from
        and filter particle based on confidance scored given from
        self.filter_particle_by_confidence.value

        Function updated ..._Prediction_Filtered Points layer.
        """
        # TODO Navya
        pass

    def _export_particles(
        self,
    ):
        """
        Fetch all positive and negative particle, and export it as .csv file
        with header [X, Y, Z, Score].

        Fetched points should be from all already labels by user or predicted.
        If positive is present score is 1 or max. confidence score from prediction
        if present. For negative prediction it should be -1 or min. confidence
        score from prediction if present.
        """
        # TODO Navya
        pass

    def _import_particles(
        self,
    ):
        """
        Import file with coordinates. Expect that files contains point in XYZ order,
        with optional confidence scores.
        Use "viridis" colormap them for the scores. If score are not present,
        assign all with score 0.
        """
        # TODO Navya
        pass
