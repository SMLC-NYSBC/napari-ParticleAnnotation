from os import rmdir
from shutil import rmtree

from fastapi.testclient import TestClient
from particleannotation.cloud.aws_api import *

from particleannotation.cloud.utils import bytes_io_to_numpy_array

client = TestClient(app)

if isdir("api/"):
    rmtree("api/")


def test_check_dir():
    response = client.get("/list_files")
    assert response.status_code == 200
    assert response.json() == []


def test_list_models():
    response = client.get("/list_models")
    assert response.status_code == 200
    assert response.json() == []


def test_new_model():
    response = client.get("/new_model")
    assert response.status_code == 200
    assert isinstance(response.json(), str)
    assert isdir("api/data/models/")
    assert isfile("api/data/models/" + response.json())


def test_upload_file():
    file = {
        "file": (
            "2d_test_data.mrc",
            open("../data/2d_test_data.mrc", "rb"),
            "image/mrc",
        )
    }

    response = client.post("/upload_file", files=file)
    assert response.status_code == 200
    assert isfile("api/data/images/2d_test_data.mrc")


def test_get_file():
    response = client.get("/get_raw_files", params={"f_name": "2d_test_data.mrc"})
    assert response.status_code == 200

    image = bytes_io_to_numpy_array(response.content)
    assert image.ndim == 2
    assert image.shape == (959, 927)
