from os import listdir, mkdir
from os.path import isdir
from typing import List

from fastapi.responses import JSONResponse
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse

from ParticleAnnotation.cloud.utils import numpy_array_to_bytes_io
from ParticleAnnotation.utils.load_data import load_image, downsample

app = FastAPI()
url = "http://3.236.214.3:8000/"
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


@app.get("/getrawfiles/")
async def list_files(f_name: str):
    try:
        image = load_image(dir_ + f_name, aws=True)
        image = downsample(image, 1 / 8)
        image = numpy_array_to_bytes_io(image)

        return StreamingResponse(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
