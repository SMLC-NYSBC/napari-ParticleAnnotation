from shutil import rmtree
from os import rmdir
from fastapi.testclient import TestClient
from ParticleAnnotation.cloud.aws_api_3d import *
from ParticleAnnotation.cloud.utils import bytes_io_to_numpy_array

client = TestClient(app)

if isdir("api/"):
    rmtree("api/")

def test_check_dir():
    response = client.get("/list_tomograms")
    assert response.status_code == 200
    assert response.json() == []

def test_upload_tomogram():
    file = {
        "file": (
            "ts0.tif",
            open("/h2/njain/original-repo/napari-ParticleAnnotation/data/ts0.tif", "rb"),
            "image/mrc",
        )
    }

    response = client.post("/upload_tomogram", files=file)
    assert response.status_code == 200
    assert isfile("api/data/tomograms/ts0.tif")

def test_upload_template():
    file = {
        "file": (
            "scores_7A4M.pt",
            open("/h2/njain/original-repo/napari-ParticleAnnotation/data/scores_7A4M.pt", "rb"),
            "image/mrc",
        )
    }

    response = client.post("/upload_template", files=file)
    assert response.status_code == 200
    assert isfile("api/data/templates/scores_7A4M.pt")

def test_get_tomogram():
    fname = "/h2/njain/original-repo/napari-ParticleAnnotation/data/ts0.tif"
    
    response = client.get("/get_tomogram", params={"f_name": fname})
    assert response.status_code == 200

    image = bytes_io_to_numpy_array(response.content)
    image_name = response.headers["X-filename"]

    assert image_name == "ts0.tif"
    assert image.ndim == 3
    assert image.shape == (250, 200, 200)

def test_get_template():
    tname = "/h2/njain/original-repo/napari-ParticleAnnotation/data/scores_7A4M.pt"

    response = client.get("/get_template", params={"f_name": tname})
    assert response.status_code == 200

    image = bytes_io_to_numpy_array(response.content)
    list_templates = response.headers["X-list_templates"].split(",")

    assert image.ndim == 3
    assert image.shape == (1, 250, 200, 200)
    assert list_templates == [tname]