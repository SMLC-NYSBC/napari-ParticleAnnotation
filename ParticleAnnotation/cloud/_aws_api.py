import json
import shutil
from os import listdir, mkdir
from os.path import isdir, isfile
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from scipy.ndimage import maximum_filter
from topaz.stats import normalize

from particleannotation.cloud.datatypes import Consensus, String, InitialValues
from particleannotation.cloud.utils import (
    numpy_array_to_bytes_io,
    get_model_name_and_weights,
)

# from particleannotation.utils.load_data import load_image, downsample
from particleannotation.utils.load_data import (
    load_template,
    load_coordinates,
    load_tomogram,
)

# from particleannotation.utils.model.active_learning_model import (
#     BinaryLogisticRegression,
#     initialize_model,
#     label_points_to_mask,
# )
from particleannotation.utils.model.active_learning_model import (
    BinaryLogisticRegression,
    label_points_to_mask,
    predict_3d_with_AL,
    stack_all_labels,
)

# from particleannotation.utils.model.utils import get_device
from particleannotation.utils.model.utils import (
    correct_coord,
    find_peaks,
    get_device,
    get_random_patch,
    rank_candidate_locations,
)
from particleannotation.utils.viewer.viewer_functionality import (
    build_gird_with_particles,
    draw_patch_and_scores,
)

app = FastAPI()
url = "http://loaclhost:8000/"
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


@app.get("/list_files", response_model=List[str])
async def list_files():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_ + "data/images/")

        return [f for f in files if f.endswith(formats)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_models", response_model=List[str])
async def list_models():
    check_dir()

    try:
        # List all files in the predefined folder
        files = listdir(dir_ + "data/models/")

        return [f for f in files if f.endswith("pth")]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/new_model", response_model=str)
async def new_model():
    # Initialize new model with weight and bias at 0.0
    model = BinaryLogisticRegression(n_features=128, l2=1.0, pi=0.01, pi_weight=1000)
    # Save model withe unique ID name
    list_model = listdir(dir_ + "data/models/")
    model_ids = [int(f[len(f) - 7 : -4]) for f in list_model if f.endswith("pth")]

    if len(model_ids) > 0:
        model_ids = model_ids[max(model_ids)] + 1
    else:
        model_ids = 0

    model_name = f"topaz_al_model_{model_ids:03}.pth"
    torch.save(model, dir_ + "data/models/" + model_name)

    return model_name


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    check_dir()

    try:
        dir_image = dir_ + "data/images/"
        file_location = f"{dir_image}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return JSONResponse(status_code=200, content={"filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_file_prediction")
async def upload_file_prediction(file: UploadFile = File(...)):
    check_dir()

    try:
        dir_image = dir_ + "data/temp/"
        file_location = f"{dir_image}/temp_prediction.tif"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        image = load_image(dir_ + "data/temp/temp_prediction.tif", aws=True)
        image = downsample(image, 1 / 8)
        image = numpy_array_to_bytes_io(image)

        return StreamingResponse(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_raw_files")
async def get_raw_files(f_name: str):
    try:
        image = load_image(dir_ + "data/images/" + f_name, aws=True)
        image = downsample(image, 1 / 8)
        image = numpy_array_to_bytes_io(image)

        return StreamingResponse(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/initialize_model_aws", response_model=List[List[float]])
async def initialize_model_aws(values: InitialValues):
    """
    Initialize model from new or pre-trained BinaryLogisticRegression class

    Notes:
        The initialize model is taken as a new model or pre-trained one. The model is
        being kept on the AWS drive and actively save/load whenever new
        training/predictions is being performed.

    Args:
        m_name (str, None): Name of the model to initialize.
        f_name (str): Image file name model is currently working on.
        n_part (int): Number of particles to generate for AL.

    Returns:
        np.array: Output generated initial points to label for the AL
    """
    m_name = values.m_name
    f_name = values.f_name
    n_part = values.n_part
    device_ = get_device()

    # Initialize temp_dir
    if isdir(dir_ + "data/temp/"):
        if (
            isfile(dir_ + "/data/temp/store_to_labels.npy")
            and isfile(dir_ + "/data/temp/x.npy")
            and isfile(dir_ + "/data/temp/y.npy")
            and isfile(dir_ + "/data/temp/count.npy")
        ):
            with open(dir_ + "/data/temp/m_name.json", "r") as openfile:
                temp_m_name = json.load(openfile)["m_name"]

            if m_name == temp_m_name:
                particle_to_label = np.load(dir_ + "/data/temp/store_to_labels.npy")
                return particle_to_label.tolist()
            else:
                shutil.rmtree(dir_ + "data/temp/")
        else:
            shutil.rmtree(dir_ + "data/temp/")
    mkdir(dir_ + "data/temp/")

    # Load image and pre-process
    image = load_image(dir_ + "data/images/" + f_name, aws=True)
    image = downsample(image, 1 / 8)

    shape = image.shape
    image, _ = normalize(image, method="gmm", use_cuda=False)

    # Compute image feature map
    x, _, particle_to_label = initialize_model(image, n_part)

    np.save(dir_ + "/data/temp/store_to_labels.npy", particle_to_label)
    np.save(dir_ + "/data/temp/consensus.npy", [])
    np.save(dir_ + "/data/temp/labeled_particles.npy", [])

    # Initialize AL model
    y = label_points_to_mask([], shape, 10)
    count = torch.where(~torch.isnan(y), torch.ones_like(y), torch.zeros_like(y))

    # Check if model exist and pick it's checkpoint
    m_name, AL_weights = get_model_name_and_weights(m_name, dir_)

    # Build model
    if AL_weights is not None:
        model = torch.load(dir_ + "data/models/" + m_name)
    else:
        model = BinaryLogisticRegression(
            n_features=x.shape[1], l2=1.0, pi=0.01, pi_weight=1000
        )

    model.fit(
        torch.from_numpy(x),
        y.ravel().to(device_),
        weights=count.ravel(),
        pre_train=AL_weights,
    )

    np.save(dir_ + "/data/temp/x.npy", x)
    np.save(dir_ + "/data/temp/y.npy", y)
    np.save(dir_ + "/data/temp/count.npy", count)
    with open(dir_ + "/data/temp/m_name.json", "w") as outfile:
        json.dump({"m_name": m_name}, outfile)
    torch.save(model, dir_ + "data/models/" + m_name)

    return particle_to_label.tolist()


@app.post("/add_pick_to_consensus")
async def add_pick_to_consensus(consensus: Consensus):
    """
    Add particles to the consensus list

    Args:
        consensus: List of corrected particles and their labels
    """
    corrected_particle = consensus.corrected_particle
    if isfile(dir_ + "data/temp/consensus.npy"):
        temp_consensus = np.load(dir_ + "data/temp/consensus.npy").tolist()
    else:
        temp_consensus = []

    temp_consensus.append(corrected_particle)
    np.save(dir_ + "data/temp/consensus.npy", temp_consensus)


@app.get("/refresh_model", response_model=list)
async def refresh_model(m_name: str, points: List[List[float]], n_part: int):
    """
    Re-trained the selected model based on checkpoint and temp data.

    Notes:
        The initialize model is loaded from the AWS drive and restored with the
        checkpoint and temp data stored in api/data/temp/.

    Args:
        m_name (str, None): Name of the model to initialize.
        points (list): List of points.
        n_part (int): Number of particles to generate for AL.

    Returns:
        np.array: Output next generated points to label for the AL
    """
    """Build consensus"""
    # Add particle to consensus

    # Build consensus

    """Add to the list of labeled particles"""
    # Marge consensus with labeled particles

    """Re-trained the model"""
    # Load model

    # Update model

    # Retrieve particles

    """Save all checkpoints"""
    np.save(dir_ + "/data/temp/store_to_labels.npy", particle_to_label)
    np.save(dir_ + "/data/temp/consensus.npy", consensus)
    np.save(dir_ + "/data/temp/labeled_particles.npy", labeled_particles)

    np.save(dir_ + "/data/temp/x.npy", x)
    np.save(dir_ + "/data/temp/y.npy", y)
    np.save(dir_ + "/data/temp/count.npy", count)
    json.dump({"m_name": m_name}, "/data/temp/m_name.json")
    torch.save(model, dir_ + "data/models/" + m_name)

    return particle_to_label.tolist()
    pass


@app.get("/retrieve_particle_to_label", response_model=List[List[float]])
async def retrieve_particle_to_label(m_name: str):
    if (
        isfile(dir_ + "/data/temp/store_to_labels.npy")
        and isfile(dir_ + "/data/temp/x.npy")
        and isfile(dir_ + "/data/temp/y.npy")
        and isfile(dir_ + "/data/temp/count.npy")
    ):
        temp_m_name = json.load("/data/temp/m_name.json")["m_name"]

        if m_name == temp_m_name:
            particle_to_label = np.load(dir_ + "/data/temp/store_to_labels.npy")
            return particle_to_label.tolist()
        else:
            return []


@app.get("/predict_topaz_al", response_model=list)
async def predict_topaz_al(model: str):
    check_dir()
    device_ = get_device()

    image = load_image(dir_ + "data/temp/temp_prediction.tif", aws=True)
    image = downsample(image, 1 / 8)
    image, _ = normalize(image, method="gmm", use_cuda=False)
    shape_ = image.shape

    weights = torch.load(dir_ + "data/models/" + model)
    AL_weights = [weights.weights, weights.bias]
    model = torch.load(dir_ + "data/models/" + model)

    x, _, _ = initialize_model(image, 1)

    model.fit(
        pre_train=AL_weights,
    )
    with torch.no_grad():
        logits = model(x).reshape(*shape_)
    logits = logits.numpy()

    max_filter = maximum_filter(logits, size=25)
    peaks = logits - max_filter
    peaks = np.where(peaks == 0)
    peaks = np.stack(peaks, axis=-1)
    if peaks.shape[1] == 3:
        peak_logits = logits[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    else:
        peak_logits = logits[peaks[:, 0], peaks[:, 1]]

    return peak_logits.tolist()
