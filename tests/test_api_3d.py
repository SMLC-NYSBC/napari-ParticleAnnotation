from shutil import rmtree
from os import rmdir
from fastapi.testclient import TestClient
from particleannotation.cloud.aws_api_3d import *
from particleannotation.cloud.utils import bytes_io_to_numpy_array

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
            open(
                "/h2/njain/original-repo/napari-particleannotation/data/ts0.tif", "rb"
            ),
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
            open(
                "/h2/njain/original-repo/napari-particleannotation/data/scores_7A4M.pt",
                "rb",
            ),
            "image/mrc",
        )
    }

    response = client.post("/upload_template", files=file, params={"tomo_name": "ts0"})
    assert response.status_code == 200
    assert isfile("api/data/templates/ts0/scores_7A4M.pt")

    file = {
        "file": (
            "scores_6R7M.pt",
            open(
                "/h2/njain/original-repo/napari-particleannotation/data/scores_6R7M.pt",
                "rb",
            ),
            "image/mrc",
        )
    }

    response = client.post("/upload_template", files=file, params={"tomo_name": "ts0"})
    assert response.status_code == 200
    assert isfile("api/data/templates/ts0/scores_6R7M.pt")


def test_get_tomogram():
    fname = "ts0.tif"

    response = client.get("/get_raw_tomos", params={"f_name": fname})
    assert response.status_code == 200

    image = bytes_io_to_numpy_array(response.content)
    image_name = response.headers["X-filename"]

    assert image_name == "ts0"
    assert image.ndim == 3
    assert image.shape == (250, 200, 200)


def test_get_template():
    tname = "ts0"

    response = client.get(
        "/get_raw_templates", params={"f_name": tname, "pdb_id": "7A4M"}
    )
    assert response.status_code == 200

    image = bytes_io_to_numpy_array(response.content)
    list_templates = response.headers["X-list_templates"].split(",")

    assert image.ndim == 4
    assert image.shape == (2, 250, 200, 200)
    assert len(list_templates) == 2
    assert list_templates == ["7A4M", "6R7M"]


def test_list_tomograms():
    response = client.get("/list_tomograms")
    assert response.status_code == 200
    assert response.json() == ["ts0"]
