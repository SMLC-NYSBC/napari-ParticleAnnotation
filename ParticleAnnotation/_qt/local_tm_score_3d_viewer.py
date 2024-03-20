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
from qtpy.QtWidgets import QFileDialog

from topaz.stats import normalize
import torch

from ParticleAnnotation.utils.load_data import downsample, load_template, load_coordinates
from ParticleAnnotation.utils.model.active_learning_model import (
    BinaryLogisticRegression,
    label_points_to_mask,
)
from ParticleAnnotation.utils.model.utils import correct_coord, find_peaks, get_device, get_random_patch


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
        self.user_annotations = np.zeros((0, 4))  # Z, Y, X, Label
        self.selected_particle_id = None
        # Remove after testing
        self.particle = None
        self.confidence = None

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

        # ---------------- Initialize New Dataset -----------------
        self.box_size = LineEdit(name="Box", value="5")
        self.patch_size = LineEdit(name="Patch", value="128")
        self.pdb_id = LineEdit(name="PDB", value='7A4M')

        self.select_particle_for_patches = PushButton(name="Select particles")
        self.select_particle_for_patches.clicked.connect(
            self._select_particle_for_patches
        )
        self.initialize_BLR = PushButton(name="Start Training")
        self.initialize_BLR.clicked.connect(self._initialize_BLR)

        # ----------- Initialize Active learning model ------------
        self.train_BLR_on_patch = PushButton(name="Change Patch")
        self.train_BLR_on_patch.clicked.connect(self._train_BLR_on_patch)

        self.show_patch = PushButton(name="Show Patch")
        self.show_patch.clicked.connect(self._show_patch)
        self.show_particle_grid = PushButton(name="Show Particle")
        self.show_particle_grid.clicked.connect(self._show_particle_grid)
        self.show_current_BLR_predictions = PushButton(name="Show BLR model")
        self.show_current_BLR_predictions.clicked.connect(self._show_current_BLR_predictions)

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
        self.export_particles = PushButton(name="Export picks")
        self.export_particles.clicked.connect(self._export_particles)
        self.import_particles = PushButton(name="Import picks")
        self.import_particles.clicked.connect(self._import_particles)

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
                        self.select_particle_for_patches,
                        self.initialize_BLR,
                    )
                ),
                HBox(
                    widgets=(
                        self.train_BLR_on_patch,
                    )
                ),
                HBox(
                    widgets=(
                        self.show_patch,
                        self.show_particle_grid,
                        self.show_current_BLR_predictions,
                    )
                ),
                HBox(widgets=(self.predict,)),
                HBox(widgets=(self.filter_particle_by_confidence,)),
                HBox(widgets=(self.export_particles, self.import_particles,)),
                HBox(widgets=(self.save_model, self.load_model,)),
            )
        )

        self.napari_viewer.window.add_dock_widget(widget, area="left")

        self.device_ = get_device()
        show_info(f"Active learning model runs on: {self.device_}")

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
        # Restart user annotation storage
        self.user_annotations = np.zeros((0, 4))

        self.image_name = self.filename = (
            self.napari_viewer.layers.selection.active.name
        )

        self.create_point_layer(
            np.array([]), np.array([]), name="Chosen_Particles_of_Interest"
        )
        self.activate_click = True

    def _initialize_BLR(
        self,
    ):
        """
        Main function which build BLR model and start first run of training

        BLR initialization:
        - Get coordinates of particles from Chosen_Particles_of_Interest
            - Store it as new self.patches list from which we will pick patch centers

        - Pre-process image and store it as self.image_preprocess

        - Build BLR model as self.model class if self.model is not None
            - If self.model is None is this mode and continue training or pre-trained model
            - Draw first patch and pre-process
            - Initialize model with model.fit
            - Use model to predict particle pick
            - Calculate entropy and select 10 particles with highest entropy

        - Present particle to the user as a grid os thing to correct
            - remove image as image layer
            - Create new image layer with particles boxes in 3D, add button to show projections

        - Wait for user correction and activation of _train_BLR_on_patch function
        """
        self.activate_click = True

        # Collect particle selected by users
        try:
            self.patches = self.napari_viewer.layers[
                "Chosen_Particles_of_Interest"
            ].data

            assert len(self.patches) > 4
            self.napari_viewer.layers.remove("Chosen_Particles_of_Interest")
        except AssertionError:
            show_info("Please choose at least 5 particles to initialize the model!")
            return

        # Image dataset pre-process
        img = self.napari_viewer.layers[self.image_name]
        self.img_process = img.data

        self.shape = self.img_process.shape
        self.img_process, _ = normalize(
            self.img_process.copy(), method="affine", use_cuda=False
        )
        self.napari_viewer.layers.remove(self.image_name)

        # Load and pre-process tm_scores data
        self.tm_scores, self.tm_idx = load_template(template=self.pdb_id.value)

        # Select patch
        self.patch_corner = get_random_patch(
                self.img_process.shape, int(self.patch_size.value), self.patches
            )

        patch = self.img_process[
                self.patch_corner[0]:self.patch_corner[0]
                + int(self.patch_size.value),
                self.patch_corner[1]:self.patch_corner[1]
                + int(self.patch_size.value),
                self.patch_corner[2]:self.patch_corner[2]
                + int(self.patch_size.value),
            ]
        self.create_image_layer(patch, name="Tomogram_Patch")

        tm_score = self.tm_scores[
            :,
            self.patch_corner[0]:self.patch_corner[0]
            + int(self.patch_size.value),
            self.patch_corner[1]:self.patch_corner[1]
            + int(self.patch_size.value),
            self.patch_corner[2]:self.patch_corner[2]
            + int(self.patch_size.value),
        ]
        self.create_image_layer(tm_score[self.tm_idx], name="TM_Scores", transparency=True)

        # Initialized y (empty label mask) and count
        self.x = torch.from_numpy(tm_score.copy()).float().permute(1, 2, 3, 0)
        self.x = self.x.reshape(-1, self.x.shape[-1])
        self.shape = patch.shape
        self.y = label_points_to_mask([], self.shape, self.box_size.value)
        self.count = torch.where(
            ~torch.isnan(self.y), torch.ones_like(self.y), torch.zeros_like(self.y)
        )

        # Initialize model
        if self.model is None:
            self.model = BinaryLogisticRegression(
                n_features=self.x.shape[1], l2=1.0, pi=0.01, pi_weight=1000
            )

        self.model.fit(
            self.x,
            self.y.ravel(),
            weights=self.count.ravel(),
            pre_train=self.AL_weights,
        )

        self.selected_particles_with_entropy, scores = find_peaks(
            tm_score[0, :], with_score=True
        )

        points = np.vstack(self.selected_particles_with_entropy[:10])
        labels = np.zeros((points.shape[0],))
        labels[:] = 2

        self.create_point_layer(points.astype(np.float64), labels)

    def _train_BLR_on_patch(
        self,
    ):
        pass

    def _predict(
        self,
    ):
        # self.particle = peaks
        # self.confidence = peak_logits
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
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File", directory="Active_learn_model.pt"
        )
        if self.AL_weights is not None:
            torch.save(self.AL_weights, filename)

    def _load_model(
        self,
    ):
        """
        Function to load and update self.AL_weights which is a list [self.weight, self.bias]
        expected as a pickle torch .pt file.

        If self.model is not None, update model weights. Else create self.model with
        this weights.
        """
        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        self.AL_weights = torch.load(f"{self.filename}")

        if self.model is not None:
            self.model.weights = self.AL_weights[0]
            self.model.bias = self.AL_weights[1]
        else:
            self.model = BinaryLogisticRegression(
                n_features=self.x.shape[1], l2=1.0, pi=0.01, pi_weight=1000
            )
            self.model.weights = self.AL_weights[0]
            self.model.bias = self.AL_weights[1]

    """
    Viewer functionality
    """

    def _show_patch(self, viewer):
        pass

    def _show_particle_grid(self, viewer):
        pass

    def _show_current_BLR_predictions(self, viewer):
        pass

    """
    Viewer helper functionality
    """

    def create_image_layer(self, image, name="TM_Scores", transparency=False):
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
            print(f"Warning: {e} error occurs while searching for {name} layer.")

        if not transparency:
            self.napari_viewer.add_image(image, name=name, colormap='gray', opacity=1.0)
        else:
            self.napari_viewer.add_image(
                image, name=name, colormap="viridis", opacity=0.25
            )

        try:
            self.napari_viewer.layers[name].contrast_limits = (
                image.min(),
                image.max(),
            )
        except Exception as e:
            print(f"Warning: {e} error occurs while searching for {name} layer.")

        if not transparency:
            # set layer as not visible
            self.napari_viewer.layers[name].visible = True
        else:
            self.napari_viewer.layers[name].visible = False

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
            self.user_annotations = np.concatenate(
                (self.user_annotations, np.hstack((points, labels[:, None])))
            )
            self.user_annotations = np.vstack(
                tuple(set(map(tuple, self.user_annotations)))
            )

            # Update point layer
            self.create_point_layer(points, labels, name)
        elif func == "remove":  # Remove point pointed by mouse
            points = point_layer.data
            labels = point_layer.properties["label"]

            # Remove point from user annotation storage
            idx = np.where(self.user_annotations[:, :3] == points[index])
            self.user_annotations = np.delete(self.user_annotations, idx, axis=0)

            # Remove point from layer
            points = np.delete(points, index, axis=0)
            labels = np.delete(labels, index, axis=0)

            # Update point layer
            self.create_point_layer(points, labels, name)
        elif func == "update":  # Update point pointed by mouse
            points = point_layer.data
            labels = point_layer.properties["label"]

            # Update point in user annotation storage
            idx = np.where(self.user_annotations[:, :3] == points[index])
            if self.user_annotations[index, -1] != label:
                self.user_annotations[index, -1] = label

            # Update point labels
            if labels[index] != label:
                labels[index] = label

                # Update point layer
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
        self.filter_particle_by_confidence.value

        Function updated ..._Prediction_Filtered Points layer.
        """
        active_layer_name = self.napari_viewer.layers.selection.active.name
        if active_layer_name.endswith("Prediction_Filtered"):
            self.napari_viewer.layers.remove(active_layer_name)
            active_layer_name = active_layer_name[:-20]
        self.napari_viewer.layers[f"{active_layer_name}"].visible = False

        if self.particle is None:
            show_info("No predicted particles to filter!")
            return
        
        keep_id = np.where(self.confidence >= self.filter_particle_by_confidence.value)

        # self.particle and self.confidence are from self._predict
        filter_particle   = self.particle[keep_id[0], :]
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

    def _export_particles(
        self,
    ):
        """
        Fetch all positive and negative particle, and export it as .csv file
        with header [Z, Y, X, Score].

        Fetched points should be from all already labels by user or predicted.
        If positive is present score is 1 or max. confidence score from prediction
        if present. For negative prediction it should be -1 or min. confidence
        score from prediction if present.

        Export self.user_annotations [n, 4] organized Z, Y, X, ID
        """ 

        # TODO - unique set of points
        
        pos_points = self.user_annotations[self.user_annotations[:, -1] == 1][:, :-1]
        # Save only user annotations (positive labels)
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save File", directory="user_annotations.csv"
        )
        data = np.hstack((pos_points, np.arange(1, pos_points.shape[0] + 1)[:, None]))
        np.savetxt(filename, data, delimiter=",", fmt="%s", header = "Z, Y, X, ID")

        pos_points = np.hstack((pos_points, np.ones((pos_points.shape[0], 1))))
        neg_points = self.user_annotations[self.user_annotations[:, -1] == 0][:, :-1]
        neg_points = np.hstack((neg_points, -1 * np.ones((neg_points.shape[0], 1))))

        data = np.vstack((pos_points, neg_points))

        # update with predicted particles
        if self.particle is not None:
            data = np.vstack((data, np.hstack((self.particle, self.confidence))))
        
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save File", directory="exported_particles.csv"
            )
            np.savetxt(filename, data, delimiter=",", fmt="%s", header="Z, Y, X, confidence")

    def _import_particles(
        self,
    ):
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
        try:
            data, labels = load_coordinates(self.filename)
            # Update user annotation storage
            self.user_annotations = np.concatenate(
                (self.user_annotations, np.hstack((data, labels[:, None])))
            )
            self.user_annotations = np.vstack(
                tuple(set(map(tuple, self.user_annotations)))
            )

            # add imported points to the layer
            self.napari_viewer.add_points(
                        data,
                        name=f"Imported_Particles",
                        properties={"confidence": labels},
                        edge_color="black",
                        face_color="confidence",
                        face_colormap="viridis",
                        edge_width=0.1,
                        symbol="disc",
                        size=5,
                    )
            show_info(f"Imported {data.shape[0]} particles!")
        except:
            show_info("Could not load coordinates!")