import json
from os import listdir, mkdir
from os.path import isdir, isfile
from typing import List

from scipy.ndimage import maximum_filter

from ParticleAnnotation.cloud.datatypes import Consensus, String, InitialValues
import numpy as np
import torch
from fastapi.responses import JSONResponse
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from topaz.stats import normalize

from ParticleAnnotation.cloud.utils import (
    numpy_array_to_bytes_io,
    get_model_name_and_weights,
)

from ParticleAnnotation.utils.load_data import (
    load_template,
    load_coordinates,
    load_tomogram,
)

from ParticleAnnotation.utils.model.active_learning_model import (
    BinaryLogisticRegression,
    label_points_to_mask,
    predict_3d_with_AL,
    stack_all_labels,
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
app = FastAPI()
url = "http://3.230.8.116:8000/"
dir_ = "api/"
formats = ("mrc", "rec", "tiff")

def check_dir():
    if not isdir("api/"):
        mkdir("api/")
    if not isdir("api/data/"):
        mkdir("api/data/")
    if not isdir("api/data/tomograms/"):
        mkdir("api/data/tomograms/")
    if not isdir("api/data/templates/"):
        mkdir("api/data/templates/")
    if not isdir("api/data/models/"):
        mkdir("api/data/models/")

@app.get("/list_tomograms", response_model=List[str])
async def list_tomograms():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_ + "data/tomograms/")

        return [f for f in files if f.endswith(formats)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/list_templates", response_model=List[str])
async def list_templates():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_ + "data/templates/")

        return [f for f in files if f.endswith(formats)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/list_models", response_model=List[str])
async def list_models():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_ + "data/models/")

        return [f for f in files if f.endswith(".pth")]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_tomogram")
async def upload_tomogram(file: UploadFile = File(...)):
    check_dir()

    try:
        dir_tomogram = dir_ + "data/tomograms/"
        file_location = f"{dir_tomogram}/{file.filename}"
        with open(file_location, "wb+") as f:
            shutil.copyfileobj(file.file, f)
        return JSONResponse(status_code=200, content={"filename": file.filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload_template")
async def upload_template(file: UploadFile = File(...)):
    check_dir()

    try:
        dir_template = dir_ + "data/templates/"
        file_location = f"{dir_template}/{file.filename}"
        with open(file_location, "wb+") as f:
            shutil.copyfileobj(file.file, f)
        return JSONResponse(status_code=200, content={"filename": file.filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO new_model, upload_file_prediction

@app.get("/get_raw_tomos")
async def get_raw_tomos(f_name: str):
    try:
        # Load the tomogram and the template
        tomogram, _, _ = load_tomogram(dir_ + "data/tomograms/" + f_name, aws = True)
        tomogram = numpy_array_to_bytes_io(tomogram)

        return StreamingResponse(tomogram)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_raw_templates")
async def get_raw_templates(f_name: str):
    try:
        # Load the tomogram and the template
        template, _ = load_template(dir_ + "data/templates/" + f_name, aws = True)
        template = numpy_array_to_bytes_io(template)

        return StreamingResponse(template)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/new_model", response_model=str)
# async def new_model():
#     # Initialize new model with weight and bias at 0.0
#     model = BinaryLogisticRegression(n_features=128, l2=1.0, pi=0.01, pi_weight=1000)
#     # Save model withe unique ID name
#     list_model = listdir(dir_ + "data/models/")
#     model_ids = [int(f[len(f) - 7 : -4]) for f in list_model if f.endswith("pth")]

#     if len(model_ids) > 0:
#         model_ids = model_ids[max(model_ids)] + 1
#     else:
#         model_ids = 0

#     model_name = f"active_learning_model_{model_ids:03}.pth"
#     torch.save(model, dir_ + "data/models/" + model_name)

#     return model_name
