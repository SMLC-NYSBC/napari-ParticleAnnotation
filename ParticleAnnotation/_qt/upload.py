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
from particleannotation.cloud._aws_api import url


class WidgetUpload(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(layout="vertical")
        self.napari_viewer = napari_viewer

        """"Widget elements"""
        spacer1 = Label(value="-- Step 1: Select and Upload data --")
        self.send_data = PushButton(name="Send Data")
        self.send_data.clicked.connect(self._send_image_to_aws)

        """Widget layer orders"""
        self.insert(1, VBox(widgets=(spacer1, self.send_data)))

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
