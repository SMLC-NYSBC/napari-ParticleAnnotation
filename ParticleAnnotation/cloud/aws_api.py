from os import listdir, mkdir, makedirs
from os.path import isdir
from typing import List

import numpy as np
from fastapi.responses import JSONResponse
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile

from ParticleAnnotation.utils.load_data import load_image

app = FastAPI()
url = "http://3.236.232.251:8000/"
dir_ = "api/data/images/"
formats = ("mrc", "rec", "tiff")


def check_dir():
    if not isdir("api/"):
        mkdir("api/")
    if not isdir("api/data/"):
        mkdir("api/data/")
    if not isdir("api/data/images/"):
        mkdir("api/data/images/")


@app.get("/listfiles/", response_model=List[str])
async def list_files():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_)

        return [f for f in files if f.endswith(formats)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    check_dir()

    try:
        file_location = f"{dir_}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return JSONResponse(status_code=200, content={"filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/getrawfiles/", response_model=np.ndarray)
async def list_files(f_name: str):
    try:
        return load_image(dir_ + f_name, aws=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
