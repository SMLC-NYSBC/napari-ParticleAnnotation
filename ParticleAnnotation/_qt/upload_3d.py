from os.path import splitext

import requests
from PyQt5.QtWidgets import QFileDialog
from magicgui.widgets import (
    Container,
    VBox,
    PushButton,
    Label,
)
from napari import Viewer
from napari.utils.notifications import show_info

from particleannotation.cloud.aws_api_3d import url


class UploadWidget(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(layout="vertical")
        self.napari_viewer = napari_viewer

        """"Widget elements"""
        spacer1 = Label(value="-- Step 1: Select and Upload Tomogram --")
        self.send_data = PushButton(name="Send Tomogram")
        self.send_data.clicked.connect(self._send_tomogram_to_aws)

        spacer2 = Label(value="-- Step 2: Select and Upload Template --")
        self.send_data_2 = PushButton(name="Send Template")
        self.send_data_2.clicked.connect(self._send_template_to_aws)

        """Widget layer orders"""
        self.insert(1, VBox(widgets=(spacer1, self.send_data)))
        self.insert(2, VBox(widgets=(spacer2, self.send_data_2)))

    def _send_tomogram_to_aws(self):
        """
        Send image file to AWS EC2 instance.
        Return:
            napari.show_info: User info prompt to indicate output.
        """

        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        root, extension = splitext(self.filename)

        format_ = extension[1:] if extension else None
        name_ = self.filename.split("/")[-1]

        print(self.filename, name_, format_)
        files = {"file": (name_, open(f"{self.filename}", "rb"), f"image/{format_}")}

        print("Sending file to AWS")
        try:
            response = requests.post(url + "upload_tomogram", files=files)

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

    def _send_template_to_aws(self):
        """
        Send template file to AWS EC2 instance.
        Return:
            napari.show_info: User info prompt to indicate output.
        """

        self.filename, _ = QFileDialog.getOpenFileName(caption="Load File")
        root, extension = splitext(self.filename)

        format_ = extension[1:] if extension else None
        name_ = self.filename.split("/")[-1]

        tomo_name = self.filename.split("/")[-2]

        files = {"file": (name_, open(f"{self.filename}", "rb"), f"image/{format_}")}

        try:
            response = requests.post(
                url + "upload_template", files=files, params={"tomo_name": tomo_name}
            )

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
