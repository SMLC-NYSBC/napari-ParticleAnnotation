from os import listdir, mkdir, makedirs
from os.path import isdir
from typing import List

from fastapi.responses import JSONResponse
import shutil
from fastapi import FastAPI, HTTPException,  File, UploadFile

app = FastAPI()
url = 'http://3.236.232.251:8000/api'
dir_ = "/api/data/images/"


@app.get("/listfiles/", response_model=List[str])
async def list_files():
    makedirs(dir_, exist_ok=True)

    try:
        # List all files in the predefined folder
        files = listdir(dir_)

        if format:
            files = [file for file in files if file.endswith(f".{format}")]
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    makedirs(dir_, exist_ok=True)

    try:
        file_location = f"{dir_}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return JSONResponse(status_code=200, content={"filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


