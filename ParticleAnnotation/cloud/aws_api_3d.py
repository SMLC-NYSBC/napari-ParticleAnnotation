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

# TODO new_model, upload_file_prediction, upload particles and import particles

@app.get("/get_raw_tomos")
async def get_raw_tomos(f_name: str):
    # Assumes 
    try:
        # Load the tomogram and the template
        tomogram, _, tomo_name = load_tomogram(dir_ + "data/tomograms/" + f_name, aws = True)
        tomogram = numpy_array_to_bytes_io(tomogram)
        headers = {"X-filename" : tomo_name}

        return StreamingResponse(tomogram, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_raw_templates")
async def get_raw_templates(f_name: str, pdb_id: str):
    try:
        # Load the tomogram and the template
        template, list_templates = load_template(dir_ + "data/templates/" + f_name + "/scores_" + pdb_id + ".pt", aws = True)
        # convert list to string
        list_templates = ",".join(map(str, list_templates))
        template = numpy_array_to_bytes_io(template)

        headers = {"X-list_templates" : list_templates}

        return StreamingResponse(template, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
