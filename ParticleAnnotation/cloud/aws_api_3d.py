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
# url = "http://3.230.8.116:8000/"
url = "http://127.0.0.1:8000"
dir_ = "api/"
formats = ("mrc", "rec", "tiff", "tif")
template_formats = ("pt", "npy")

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

def check_dir_tomo(tomo_name):
    if not isdir("api/data/templates/" + tomo_name):
        mkdir("api/data/templates/" + tomo_name)

@app.get("/list_tomograms", response_model=List[str])
async def list_tomograms():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_ + "data/tomograms/")
        files = [f for f in files if f.endswith(formats)]
        files = [f.split(".")[0] for f in files]

        return files

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/list_templates", response_model=List[str])
async def list_templates():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_ + "data/templates/")
        files = [f for f in files if f.endswith(template_formats)]

        return files

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
async def upload_template(file: UploadFile = File(...), tomo_name: str = None):
    check_dir()
    check_dir_tomo(tomo_name)

    try:
        dir_template = dir_ + "data/templates/" + tomo_name
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
    """
        Get the template of the tomogram
        This assumes that the template is under the folder data/templates/tomo_name and 
        file name is of the format scores_pdb_id.pt or tardis_6QS9.pt.

        Loads all the templates in the folder data/templates/tomo_name with extension .pt.
    """
    try:
        # Load the tomogram and the template
        if pdb_id == "6QS9":
            template, list_templates = load_template(dir_ + "data/templates/" + f_name + "/tardis_6QS9.pt")
        else:
            template, list_templates = load_template(dir_ + "data/templates/" + f_name + "/scores_" + pdb_id + ".pt")
        # convert list to string
        list_templates = ",".join(map(str, list_templates))
        template = numpy_array_to_bytes_io(template)

        headers = {"X-list_templates" : list_templates}

        return StreamingResponse(template, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
