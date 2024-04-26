import requests
from imageio import imwrite

from magicgui.widgets import (
    Container,
    VBox,
    PushButton,
    Label,
    FloatSlider,
    ComboBox,
)
from napari.utils.notifications import show_info
from napari import Viewer

from particleannotation.cloud.aws_api import url
from particleannotation.cloud.utils import bytes_io_to_numpy_array


class AnnotationWidgetv2(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(layout="vertical")
        self.napari_viewer = napari_viewer

        """"Widget elements"""
        spacer1 = Label(value="-- Step 1: Select pre-train model --")
        self.load_ALM = ComboBox(name="Select Model", choices=self._update_model_list())
        self.send_data = PushButton(name="Upload file for prediction")
        self.send_data.clicked.connect(self._send_image_to_aws)

        spacer2 = Label(value="---------------- Step 2: Predict ----------------")
        self.predict = PushButton(name="Predict")
        self.predict.clicked.connect(self._predict)

        spacer3 = Label(value="--------- Step 3: Visualize labels tool ---------")
        self.slide_pred = FloatSlider(
            name="Filter Particle",
            min=0,
            max=1,
        )
        self.slide_pred.changed.connect(self._filter_particle)

        self.reset_view = PushButton(name="Reset View")
        self.reset_view.clicked.connect(self._reset_view)

        """Widget layer orders"""
        self.insert(
            1,
            VBox(
                widgets=(
                    spacer1,
                    self.load_ALM,
                    spacer2,
                    self.send_data,
                    self.predict,
                    spacer3,
                    self.slide_pred,
                    self.reset_view,
                )
            ),
        )

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

    def _refresh(self):
        pass

    def _reset_view(self):
        self.load_ALM.choices = self._update_model_list()
        self.load_data.choices = self._update_data_list()
        self.napari_viewer.reset_view()

    def _send_image_to_aws(self):
        """
        Send image file to AWS EC2 instance.

        Return:
            napari.show_info: User info prompt to indicate output.
        """
        # Package image file in correct format
        active_layer_name = self.napari_viewer.layers.selection.active.name
        self.image_layer_name = active_layer_name
        img = self.napari_viewer.layers[active_layer_name]
        imwrite("temp_prediction.tif", img)
        format_ = "tif"
        name_ = "temp_prediction"

        files = {"file": (name_, open("temp_prediction.tif", "rb"), f"image/{format_}")}

        # Call API for response
        try:
            response = requests.get(url + "upload_file_prediction", files=files)

            if response.status_code == 200:
                show_info(f"File uploaded successfully: {response.json()}")

                image = bytes_io_to_numpy_array(response.content)
                self.napari_viewer.add_image(data=image, name="Raw_Image")
                self.napari_viewer.layers["Raw_Image"].contrast_limits = (
                    image.min(),
                    image.max(),
                )
                self._reset_view()
            else:
                show_info(
                    "Failed to upload file: \n"
                    f"Error: {response.status_code} \n"
                    f"Message:{response.text}"
                )
        except:
            show_info(f"Connection Error to {url}. Check if server is running.")

    def _predict(self):
        params = {"model": self.load_ALM.value}
        try:
            response = requests.get(url + "predict_topaz_al", json=params)

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

    def _filter_particle(self):
        pass
