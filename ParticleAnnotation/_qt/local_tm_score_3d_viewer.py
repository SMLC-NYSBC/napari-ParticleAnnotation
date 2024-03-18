import numpy as np
from magicgui.widgets import (
    Container,
    PushButton,
    LineEdit,
    FloatSlider,
    create_widget,
    SpinBox,
    VBox,
    HBox,
)
from napari import Viewer
from napari.layers import Points
from napari.utils.notifications import show_info
from napari.settings import get_settings

from ParticleAnnotation.utils.model.utils import get_device


class AnnotationWidget(Container):
    def __init__(self, viewer_tm_score_3d: Viewer):
        super().__init__(layout="vertical")
        settings = get_settings()
        settings.appearance.theme = 'dark'

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
            self.napari_viewer.bind_key("z", self.ZEvent)  # Add/Update to Negative label
            self.napari_viewer.bind_key("x", self.XEvent)  # Add/Update to Positive label
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

        self.napari_viewer.window.add_dock_widget(
                    widget, area="left"
                )

        device_ = get_device()
        show_info(f"Active learning model runs on: {device_}")

    """
    Mouse and keys bindings
    """

    def track_mouse_position(self, viewer, event):
        self.mouse_position = event.position

    def selected_point_near_mouse(self, viewer, event):
        if self.activate_click:
            try:
                # if self.activate_click:
                points_layer = viewer.layers["Initial_Labels"].data

                # TODO Filter points_layer and search only for points withing radius
                # TODO Just in case we have thousands or millions of points issue

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

    def ZEvent(
        self,
    ):
        pass

    def XEvent(
        self,
    ):
        pass

    def CEvent(
        self,
    ):
        pass

    """
    Main triggers for GUI
    """

    def _select_particle_for_patches(
        self,
    ):
        pass

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
    Helper functions
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
        # TODO Navya
        pass

    def _export_particles(
        self,
    ):
        pass

    def _import_particles(
        self,
    ):
        pass
