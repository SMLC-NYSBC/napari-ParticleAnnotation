from os import listdir, mkdir, rmdir
from os.path import isdir
from typing import List, Union

import numpy as np
import torch
from fastapi.responses import JSONResponse
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from topaz.stats import normalize

from ParticleAnnotation.cloud.utils import numpy_array_to_bytes_io
from ParticleAnnotation.utils.load_data import load_image, downsample
from ParticleAnnotation.utils.model.active_learning_model import BinaryLogisticRegression, initialize_model, \
    label_points_to_mask
from ParticleAnnotation.utils.model.utils import get_device

app = FastAPI()
url = "http://3.230.8.116:8000/"
dir_ = "api/"
formats = ("mrc", "rec", "tiff")


def check_dir():
    if not isdir("api/"):
        mkdir("api/")
    if not isdir("api/data/"):
        mkdir("api/data/")
    if not isdir("api/data/images/"):
        mkdir("api/data/images/")

    if not isdir("api/data/models/"):
        mkdir("api/data/models/")


@app.get("/list_files/", response_model=List[str])
async def list_files():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_+"data/images/")

        return [f for f in files if f.endswith(formats)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_models/", response_model=List[str])
async def list_models():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_+"data/models/")

        return [f for f in files if f.endswith("pth")]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    check_dir()

    try:
        dir_image = dir_+"data/images/"
        file_location = f"{dir_image}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return JSONResponse(status_code=200, content={"filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_raw_files/")
async def get_raw_files(f_name: str):
    try:
        image = load_image(dir_ + "data/images/" + f_name, aws=True)
        image = downsample(image, 1 / 8)
        image = image.astype(np.int8)
        image = numpy_array_to_bytes_io(image)

        return StreamingResponse(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/new_model/")
async def new_model():
    # Initialize new model with weight and bias at 0.0
    model = BinaryLogisticRegression(
        n_features=128, l2=1.0, pi=0.01, pi_weight=1000
    )
    # Save model withe unique ID name
    list_model = listdir(dir_+"data/models/")
    model_ids = [int(f[len(f)-7:-4]) for f in list_model if f.endswith("pth")]
    model_ids = model_ids[max(model_ids)] + 1

    # ToDo output 000 values
    model_name = f"topaz_al_model_{model_ids:03}.pth"

    torch.save(model, dir_ + "data/models/" + model_name)
    return model_name


@app.post("/initialize_model/")
async def initialize_model(m_name: Union[str, None], f_name: str):
    # Initialize temp_dir
    if isdir(dir_+"data/temp/"):
        rmdir(dir_ + "data")
    mkdir(dir_+"data/temp/")

    # Load image and pre-process
    image = load_image(dir_ + "data/images/" + f_name, aws=True)
    image = downsample(image, 1 / 8)

    shape = image.shape
    image, _ = normalize(image, method="gmm", use_cuda=False)

    # Compute image feature map
    x, _, p_label = initialize_model(image)

    # Initialize AL model
    y = label_points_to_mask([], shape, 10)
    count = torch.where(
        ~torch.isnan(y), torch.ones_like(y), torch.zeros_like(y)
    )

    # Check if model exist and pick it's checkpoint
    list_model = listdir(dir_ + "data/models/")
    model_ids = [int(f[len(f) - 7:-4]) for f in list_model if f.endswith("pth")]

    if m_name in model_ids or m_name is None:
        AL_weights = torch.load(dir_ + "data/models/" + m_name)
        AL_weights = [AL_weights.weight, AL_weights.bias]
    else:
        AL_weights = None

    # Build model
    model = BinaryLogisticRegression(
        n_features=x.shape[1], l2=1.0, pi=0.01, pi_weight=1000
    )
    model.fit(
        x,
        y.ravel(),
        weights=count.ravel(),
        pre_train=AL_weights,
    )

    torch.save(model, dir_ + "data/models/" + m_name)
