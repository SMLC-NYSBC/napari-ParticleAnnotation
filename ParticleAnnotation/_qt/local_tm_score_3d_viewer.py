import numpy as np
from magicgui.widgets import (
    Container,
    PushButton,
    LineEdit,
    FloatSlider,
    Label,
    VBox,
    HBox,
)
from napari import Viewer
from napari.utils.notifications import show_info
from sklearn.neighbors import KDTree
from qtpy.QtWidgets import QFileDialog
import tifffile.tifffile as tif

from topaz.stats import normalize
import torch

from ParticleAnnotation.utils.load_data import (
    load_template,
    load_coordinates,
    load_tomogram,
)
from ParticleAnnotation.utils.model.active_learning_model import (
    BinaryLogisticRegression,
    label_points_to_mask,
    predict_3d_with_AL,
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


class AnnotationWidget(Container):
    def __init__(self, viewer_tm_score_3d: Viewer):
        """
        ToDo
            - Patch over sized
            - predicted particles changes colors
            - marge initialize blr to re-train
            - make a clear gui
                - Load data, Train, Predict, Utils
        """
        super().__init__(layout="vertical")
        self.napari_viewer = viewer_tm_score_3d

        # Global
        self.image_name = ""
        self.filename = None
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

        spacer_1 = Label(value="------------- Options -------------")
        self.box_size = LineEdit(name="Mask Size", value="10")
        self.filter_size = LineEdit(name="Filter Size", value="15")
        self.patch_size = LineEdit(name="Patch size", value="128")
        self.pdb_id = LineEdit(name="PDB", value="7A4M")

        # ---------------- Import & Export modules ----------------
        self.export_particles = PushButton(name="Save particles")
        self.export_particles.clicked.connect(self._export_particles)
        self.import_particles = PushButton(name="Load particles")
        self.import_particles.clicked.connect(self._import_particles)

        self.save_model = PushButton(name="Save Model")
        self.save_model.clicked.connect(self._save_model)
        self.load_model = PushButton(name="Load Model")
        self.load_model.clicked.connect(self._load_model)

        spacer_2 = Label(value="------------ Training -------------")
        self.select_particle_for_patches = PushButton(name="Start Training")
        self.select_particle_for_patches.clicked.connect(
            self._select_particle_for_patches
        )
        self.train_BLR_on_patch = PushButton(name="Re-train model")
        self.train_BLR_on_patch.clicked.connect(self._train_BLR_on_patch)

        spacer_4 = Label(value="---------- Viewing modes ----------")
        self.show_patch = PushButton(name="Tomogram")
        self.show_patch.clicked.connect(self._show_patch)
        self.show_particle_patch_grid = PushButton(name="Correct Particle")
        self.show_particle_patch_grid.clicked.connect(self._show_particle_patch_grid)
        self.show_particle_all_grid = PushButton(name="All Particle")
        self.show_particle_all_grid.clicked.connect(self._show_particle_all_grid)
        # self.show_current_BLR_predictions = PushButton(name="BLR model")
        # self.show_current_BLR_predictions.clicked.connect(
        #     self._show_current_BLR_predictions
        # )

        spacer_3 = Label(value="------------ Predict --------------")
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

        widget = VBox(
            widgets=(
                spacer_1,
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
                HBox(
                    widgets=(
                        self.export_particles,
                        self.import_particles,
                    )
                ),
                HBox(
                    widgets=(
                        self.save_model,
                        self.load_model,
                    )
                ),
                spacer_2,
                HBox(
                    widgets=(
                        self.select_particle_for_patches,
                        self.train_BLR_on_patch,
                    )
                ),
                spacer_3,
                HBox(widgets=(self.predict,)),
                HBox(widgets=(self.filter_particle_by_confidence,)),
                spacer_4,
                HBox(
                    widgets=(
                        self.show_patch,
                        self.show_particle_patch_grid,
                        self.show_particle_all_grid,
                        # self.show_current_BLR_predictions,
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
                pass

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

    """""" """""" """""" """
    Main triggers for GUI
    """ """""" """""" """"""

    def _select_particle_for_patches(
        self,
    ):
        """
        Starting function - for the plugin. Function is allowing user to label
        particles of interest, and store them.
        """
        filter_size = int(self.filter_size.value)

        # If image is not loaded, ask user to load it
        if self.napari_viewer.layers.selection.active is None:
            img, _, name = load_tomogram()
            self.create_image_layer(img, name=name, transparency=False)

        self.image_name = self.filename = (
            self.napari_viewer.layers.selection.active.name
        )
        img = self.napari_viewer.layers[self.image_name]
        self.img_process = img.data
        self.img_process, _ = normalize(
            self.img_process.copy(), method="affine", use_cuda=False
        )

        # Load and pre-process tm_scores data
        self.tm_scores, self.tm_idx = load_template(template=self.pdb_id.value)
        if self.tm_scores is None:
            show_info(f"Please select template score for {self.pdb_id.value}")
            return

        # Restart user annotation storage
        self.user_annotations = np.zeros((0, 4))

        self.create_image_layer(
            self.tm_scores[self.tm_idx], name="TM_Scores", transparency=True
        )

        peaks, _ = find_peaks(self.tm_scores[self.tm_idx], filter_size, with_score=True)

        peaks = np.concatenate((peaks[:10, :], peaks[-10:, :]))
        peaks = np.hstack((peaks, np.zeros((20, 1))))
        peaks[10:, 3] = 1
        self.user_annotations = peaks

        # Pre-select top 10 and lowest 10 as particle proposals
        self.create_point_layer(
            peaks[:, :3], peaks[:, 3], name="Chosen_Particles_of_Interest"
        )
        self._show_particle_all_grid()
        self.activate_click = True

    def _train_BLR_on_patch(
        self,
    ):
        self.all_grid = False
        self.grid_labeling_mode = False

        patch_size = int(self.patch_size.value)
        """If Patch do not exist create one and drawn panicles"""
        if self.patch_corner is None:
            self.napari_viewer.layers.remove(self.image_name)

            self.patch_corner = get_random_patch(
                self.img_process.shape, patch_size, self.user_annotations[:, :3]
            )

        self.clean_viewer()
        patch, tm_score = draw_patch_and_scores(
            self.img_process, self.tm_scores, self.patch_corner, patch_size
        )

        # Initialized y (empty label mask) and count
        self.x = torch.from_numpy(tm_score.copy()).float().permute(1, 2, 3, 0)
        self.x = self.x.reshape(-1, self.x.shape[-1])
        self.shape = patch.shape

        if self.model is None:
            self.model = BinaryLogisticRegression(
                n_features=self.x.shape[0], l2=1.0, pi=0.01, pi_weight=1000
            )

        """Re-trained BLR model on user corrected particles"""
        # Retrive coordinate within patch
        stored_points = self.user_annotations.copy()[:, :3] - self.patch_corner
        point_indexes = np.all(
            (stored_points >= 0) & (stored_points <= patch_size), axis=1
        )
        points = correct_coord(
            self.user_annotations[point_indexes, :3], self.patch_corner, False
        )
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
        print(data)
        self.y = label_points_to_mask(data, self.shape, self.box_size.value)
        self.count = (~torch.isnan(self.y)).float()

        # Re-trained BLR model
        self.model.fit(self.x, self.y.ravel(), weights=self.count.ravel())

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
        self.x = torch.from_numpy(tm_score.copy()).float().permute(1, 2, 3, 0)
        self.x = self.x.reshape(-1, self.x.shape[-1])

        with torch.no_grad():
            logits = self.model(self.x).reshape(*self.shape)
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
        self.patch_points = np.vstack((self.patch_points, stored_points[point_indexes]))
        self.patch_label = np.hstack(
            (self.patch_label, self.user_annotations[point_indexes, 3])
        )

        """Display new particles and image layers"""
        self.create_image_layer(patch, name="Tomogram_Patch")
        self.create_image_layer(
            tm_score[self.tm_idx], name="TM_Scores", transparency=True
        )
        self.create_point_layer(
            self.patch_points, self.patch_label, name="Particle_BLR_is_Uncertain"
        )

        self._show_particle_patch_grid()
        self.napari_viewer.reset_view()

    def _predict(
        self,
    ):
        self.all_grid = False
        self.grid_labeling_mode = False
        self.clean_viewer()

        if self.model is None:
            return

        patch_size = int(self.patch_size.value)

        peaks, peaks_confidence, logits = predict_3d_with_AL(
            self.img_process,
            tm_scores=self.tm_scores,
            model=self.model,
            offset=patch_size,
            maximum_filter_size=15,  # Make it user parameter
        )
        order = np.argsort(peaks_confidence)
        peaks = peaks[order]
        peaks_confidence = peaks_confidence[order]

        self.create_image_layer(logits, "Logits", transparency=True)
        tif.imwrite("logits.tif", logits)
        tif.imwrite("tm_scores.tif", self.tm_scores[self.tm_idx])

        self.napari_viewer.add_points(
            peaks,
            name="Particle_Prediction",
            properties={"label": peaks_confidence},
            edge_color="black",
            face_color="label",
            face_colormap="Spectral",
            edge_contrast_limits=(0, 1),
            edge_width=0.1,
            symbol="disc",
            size=5,
        )
        self._filter_particle_by_confidence()

    """""" """""" """""" """""" """
    BLR Model helper functions
    """ """""" """""" """""" """"""

    def _save_model(
        self,
    ):
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
            self.model.fit(pre_train=self.AL_weights)
        else:
            self.model = BinaryLogisticRegression(
                n_features=self.AL_weights[0].shape[0], l2=1.0, pi=0.01, pi_weight=1000
            )
            self.model.fit(pre_train=self.AL_weights)

        self.AL_weights = None  # Reset to allow re-training

    """""" """""" """""" """
    Viewer functionality
    """ """""" """""" """"""

    def _show_patch(self):
        """
        Viewer function to display a current patch and all particles in it.
        """
        self.all_grid = False
        self.grid_labeling_mode = False
        self.clean_viewer()

        patch_size = int(self.patch_size.value)
        if self.img_process is not None:
            patch, _ = draw_patch_and_scores(
                self.img_process, self.tm_scores, self.patch_corner, patch_size
            )
            self.create_image_layer(patch, name="Tomogram_Patch")

        if self.tm_scores is not None:
            _, tm_score = draw_patch_and_scores(
                self.img_process, self.tm_scores, self.patch_corner, patch_size
            )
            self.create_image_layer(
                tm_score[self.tm_idx], name="TM_Scores", transparency=True
            )

        if self.patch_points is not None:
            self.create_point_layer(
                self.patch_points, self.patch_label, name="Particle_BLR_is_Uncertain"
            )

    def _show_particle_patch_grid(self):
        """
        Viewer function to show all stored particles in current patch based on
        self.user_annotation and particle_layer. Particles are shown as a grid
        with particle in the center.
        """
        self.all_grid = False
        self.grid_labeling_mode = True
        self.clean_viewer()

        (
            crop_grid_img,
            crop_grid_tm_scores,
            grid_particle_points,
            grid_particle_labels,
        ) = build_gird_with_particles(
            self.patch_points,
            self.patch_label,
            self.patch_corner,
            self.img_process,
            self.tm_scores,
            self.tm_idx,
            int(self.box_size.value),
        )

        self.create_image_layer(crop_grid_img, name="Particles_crops")
        self.create_image_layer(
            crop_grid_tm_scores, name="Particles_crops_scores", transparency=True
        )
        self.create_point_layer(
            grid_particle_points, grid_particle_labels, "Particle_BLR_is_Uncertain"
        )

    def _show_particle_all_grid(self):
        """
        Viewer function to show all stored particles on self.user_annotation
        and particle_layer. Particles are shown as a grid with particle in the center.
        """
        self.all_grid = True
        self.grid_labeling_mode = True
        self.clean_viewer()

        (
            crop_grid_img,
            crop_grid_tm_scores,
            grid_particle_points,
            grid_particle_labels,
        ) = build_gird_with_particles(
            self.user_annotations[:, :3],
            self.user_annotations[:, 3],
            (0, 0, 0),
            self.img_process,
            self.tm_scores,
            self.tm_idx,
            int(self.box_size.value),
        )

        self.create_image_layer(crop_grid_img, name="Particles_crops")
        self.create_image_layer(
            crop_grid_tm_scores, name="Particles_crops_scores", transparency=True
        )
        self.create_point_layer(
            grid_particle_points, grid_particle_labels, "Particle_BLR_is_Uncertain"
        )

    def _show_current_BLR_predictions(self):
        """
        Viewer function to run current BLR model and show predicted particles on
        the current patch.
        """
        self.all_grid = False
        self.grid_labeling_mode = False
        self.clean_viewer()

        patch_size = int(self.patch_size.value)
        if self.model is not None:
            patch, tm_score = draw_patch_and_scores(
                self.img_process, self.tm_scores, self.patch_corner, patch_size
            )
            self.create_image_layer(patch, name="Tomogram_Patch")
            self.create_image_layer(
                tm_score[self.tm_idx], name="TM_Scores", transparency=True
            )

            # Features from current patch
            self.shape = patch.shape
            self.x = torch.from_numpy(tm_score.copy()).float().permute(1, 2, 3, 0)
            self.x = self.x.reshape(-1, self.x.shape[-1])

            with torch.no_grad():
                logits = self.model(self.x).reshape(*self.shape)
                logits = logits.cpu().detach().numpy()

            self.create_image_layer(logits, "Logits", True)

            blr_model_state_points, blr_model_state_labels = find_peaks(
                logits, 15, True
            )

            blr_model_state_points = blr_model_state_points[-25:, :]
            blr_model_state_labels = blr_model_state_labels[-25:, :].flatten()
            self.napari_viewer.add_points(
                blr_model_state_points,
                name="BLR_prediction",
                properties={"label": blr_model_state_labels},
                edge_color="black",
                face_color="label",
                face_colormap="Spectral",
                edge_width=0.1,
                symbol="disc",
                size=5,
            )
        else:
            self._show_patch()
            show_info("Warning No BLR model load!")

    """""" """""" """""" """""" """
    Viewer helper functionality
    """ """""" """""" """""" """"""

    def clean_viewer(
        self,
    ):
        self.napari_viewer.layers.select_all()
        self.napari_viewer.layers.remove_selected()

    def create_image_layer(
        self, image, name="TM_Scores", transparency=False, viability=True
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
            show_info(f"Warning: {e}. Layer {name} does not exist, creating new one.")

        if transparency:
            self.napari_viewer.add_image(
                image, name=name, colormap="Spectral", opacity=0.5
            )
        else:
            self.napari_viewer.add_image(image, name=name, colormap="gray", opacity=1.0)

        try:
            self.napari_viewer.layers[name].contrast_limits = (
                image.min(),
                image.max(),
            )
        except Exception as e:
            pass

        if viability:
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
            self.napari_viewer.layers[name].mode = "select"
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

    def update_point_grid(self, index=None, label=0, func="update"):
        point_layer = self.napari_viewer.layers["Particle_BLR_is_Uncertain"]
        points = point_layer.data
        labels = point_layer.properties["label"]

        if self.all_grid:
            point = correct_coord(self.patch_points[index], (0, 0, 0), True)
        else:
            point = correct_coord(self.patch_points[index], self.patch_corner, True)

        if func == "update":
            # Update point labels
            if labels[index] != label:
                labels[index] = label

            self.patch_label = labels

            # Check if point index from self.point_layer is already in self.user_annotations
            # Add and/or update point label in self.user_annotations
            idx = self.user_annotations[:, :3] - point
            idx = 0 in np.sum(idx.astype(np.float16), axis=1)

            if idx:  # Add point to self.user_annotation
                idx = np.where(point in self.user_annotations[:, :3])
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
            points = np.delete(self.patch_points, index, axis=0)
            self.patch_points = np.delete(self.patch_points, index, axis=0)
            self.patch_label = np.delete(self.patch_label, index, axis=0)
            labels = np.delete(self.patch_points, index, axis=0)

            idx = point in self.user_annotations[:, :3]
            if idx:  # Remove point to self.user_annotation
                idx = np.where(point in self.user_annotations[:, :3])[0][0]

                self.user_annotations = np.delete(self.user_annotations, index, axis=0)

        self.create_point_layer(points, labels, "Particle_BLR_is_Uncertain")

    def update_point_layer(self, index=None, label=0, func="add"):
        name = self.napari_viewer.layers.selection.active.name
        point_layer = self.napari_viewer.layers[name]

        try:
            point_layer = self.napari_viewer.layers["Particle_BLR_is_Uncertain"]
            self.patch_points = point_layer.data
            self.patch_label = point_layer.properties["label"]
        except Exception as e:
            pass

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

            # Update point layer
            self.create_point_layer(points, labels, name)
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

            # Update point layer
            self.create_point_layer(points, labels, name)
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

    """""" """""" """""" """""
    Global helper functions
    """ """""" """""" """""" ""

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

            self.napari_viewer.add_points(
                particles_filter,
                name="Particle_Prediction_Filtered",
                properties={"label": confidence_filter},
                edge_color="black",
                face_color="label",
                face_colormap="Spectral",
                edge_width=0.1,
                face_contrast_limits=(0, 1),
                symbol="disc",
                size=5,
            )
        except Exception as e:
            pass

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
            face_colormap="Spectral",
            edge_width=0.1,
            symbol="disc",
            size=5,
        )
        show_info(f"Imported {data.shape[0]} particles!")
