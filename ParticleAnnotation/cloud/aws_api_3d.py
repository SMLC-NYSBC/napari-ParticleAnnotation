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
    stack_all_labels,
)
from ParticleAnnotation.utils.model.utils import correct_coord, find_peaks, rank_candidate_locations
from ParticleAnnotation.utils.viewer.viewer_functionality import draw_patch_and_scores

app = FastAPI()
url = "http://localhost:8000/"
dir_ = ""
formats = ("mrc", "rec", "tiff", "tif")
template_formats = ("pt", "npy")

"""
Initialization of the plugin
"""


def template_list(f_name):
    # Load TM Scores
    files = listdir("data/" + f_name)
    files = [f[:-3] for f in files if f.endswith(".pt")]
    ice_ = [True if i.endswith("scores_ice") else False for i in files]
    if sum(ice_) > 0:
        ice_ = files[np.where(ice_)[0][0]]
        files.remove(ice_)
        files.append(ice_)

    return [f"data/{f_name}/{s}.pt" for s in files]


@app.get("/list_tomograms", response_model=List[str])
async def list_tomograms():
    try:
        # List all files in the predefined folder
        files = listdir("data/")
        files = [f for f in files if f.startswith("T_")]

        return files

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_templates", response_model=List[str])
async def list_templates(tomo_name: str = None):
    try:
        template = template_list(tomo_name)
        template = [t.split("/")[-1][:-3] for t in template]

        return template

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/upload_tomogram")
# async def upload_tomogram(file: UploadFile = File(...)):
#     try:
#         dir_tomogram = dir_ + "data/tomograms/"
#         file_location = f"{dir_tomogram}/{file.filename}"
#         with open(file_location, "wb+") as f:
#             shutil.copyfileobj(file.file, f)
#         return JSONResponse(status_code=200, content={"filename": file.filename})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/upload_template")
# async def upload_template(file: UploadFile = File(...), tomo_name: str = None):
#     try:
#         dir_template = dir_ + "data/templates/" + tomo_name
#         file_location = f"{dir_template}/{file.filename}"
#         with open(file_location, "wb+") as f:
#             shutil.copyfileobj(file.file, f)
#         return JSONResponse(status_code=200, content={"filename": file.filename})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # TODO new_model, upload_file_prediction, upload particles and import particles

"""
Retrive data for the visualization
"""


@app.get("/get_raw_tomos")
async def get_raw_tomos(f_name: str):
    try:
        # Load the tomogram and the template
        tomogram, _, tomo_name = load_tomogram(
            "data/" + f_name + f"/{f_name}.mrc", aws=True
        )

        min_ = tomogram.min()
        max_ = tomogram.max()
        tomogram = ((tomogram - min_) / (max_ - min_)) * 128
        tomogram = tomogram.astype(np.int8)

        tomogram = numpy_array_to_bytes_io(tomogram)
        headers = {"X-filename": tomo_name}

        return StreamingResponse(tomogram, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_raw_templates")
async def get_raw_templates(f_name: str, pdb_name: str):
    """
    Get the template of the tomogram
    This assumes that the template is under the folder data/templates/tomo_name and
    file name is of the format scores_pdb_id.pt or tardis_6QS9.pt.

    Loads all the templates in the folder data/templates/tomo_name with extension .pt.
    """
    pdb_name = pdb_name.split("|")

    try:
        pdb_name = [f"data/{f_name}/{f}.pt" for f in pdb_name]

        # Load the tomogram and the template
        template = load_template(pdb_name, aws=True)

        min_ = template.min()
        max_ = template.max()
        template = ((template - min_) / (max_ - min_)) * 128
        template = template.astype(np.int8)

        # convert list to string
        template = numpy_array_to_bytes_io(template)
        headers = {"X-list_templates": "TM_Scores"}

        return StreamingResponse(template, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
Get first particles based on TM scores
"""


@app.get("/get_initial_peaks")
async def get_initial_peaks(f_name: str, filter_size: int, tm_idx: int):
    pdb_name = template_list(f_name)

    template, _ = load_template(pdb_name)

    # Generate Peaks
    peaks, _ = find_peaks(template[tm_idx], filter_size, with_score=True)
    peaks = peaks[-20:, :]
    peaks = np.hstack((peaks, np.zeros((20, 1))))
    peaks[:, 3] = 2

    peaks_ice, _ = find_peaks(template[-1], filter_size, with_score=True)
    peaks_ice = peaks_ice[-3:, :]
    peaks_ice = np.hstack((peaks_ice, np.zeros((3, 1))))
    peaks_ice[:, 3] = 2

    peaks = np.vstack((peaks, peaks_ice))

    peaks_ice, _ = find_peaks(template[-2], filter_size, with_score=True)
    peaks_ice = peaks_ice[-3:, :]
    peaks_ice = np.hstack((peaks_ice, np.zeros((3, 1))))
    peaks_ice[:, 3] = 2

    peaks = np.vstack((peaks, peaks_ice))

    peaks_ice, _ = find_peaks(template[-3], filter_size, with_score=True)
    peaks_ice = peaks_ice[-4:, :]
    peaks_ice = np.hstack((peaks_ice, np.zeros((4, 1))))
    peaks_ice[:, 3] = 2

    # Sent numpy array
    peaks = np.vstack((peaks, peaks_ice))

    peaks = numpy_array_to_bytes_io(peaks)
    headers = {"X-list_templates": "Peaks"}

    return StreamingResponse(peaks, headers=headers)


@app.get("/re_train_model")
async def re_train_model(
    f_name: str,
    tm_idx: int,
    patch_corner: str,
    patch_size: int,
    box_size: int,
    pi: float,
    weights: str,
    bias: str,
    points: str,
):
    patch_corner = patch_corner.split(",")
    patch_corner = tuple(map(int, patch_corner))

    if weights == "":
        weights = None
    else:
        weights = weights.split(",")
        weights = tuple(map(float, weights))
        weights = torch.from_numpy(np.array(weights).astype(np.float32))

    if bias == "":
        bias = None
    else:
        bias = float(bias)
        bias = torch.Tensor([bias])

    points = np.array(tuple(map(float, points.split(","))))
    points = points.reshape((points.shape[0] // 4, 4))

    tomogram, _, _ = load_tomogram("data/" + f_name + f"/{f_name}.mrc", aws=True)

    pdb_name = template_list(f_name)
    template, _ = load_template(pdb_name)

    _, tm_score = draw_patch_and_scores(
        tomogram, template, patch_corner, patch_size
    )

    x = torch.from_numpy(tm_score.copy()).float()
    x = x.permute(1, 2, 3, 0)
    x = x.reshape(-1, x.shape[-1])
    shape = tm_score.shape[1:]

    model = BinaryLogisticRegression(
        n_features=x.shape[-1],
        l2=1.0,
        pi=float(pi),
        pi_weight=1000,
    )
    if weights is not None:
        model.fit(pre_train=[weights, bias])

    stored_points = points.copy()[:, :3] - patch_corner
    point_indexes = np.all((stored_points >= 0) & (stored_points <= patch_size), axis=1)
    point = points[point_indexes, ...]

    # Update BLR inputs
    data = np.hstack((point[:, 3][:, None], point[:, :3]))

    data[:, 1:] = correct_coord(data[:, 1:], patch_corner, False)
    y = label_points_to_mask(data, shape, box_size)
    count = (~torch.isnan(y)).float()
    count[y == 0] = 0

    data = np.hstack((points[:, 3][:, None], points[:, :3]))

    all_labels = stack_all_labels(template, data, box_size)

    if len(all_labels[0][0]) > 0:
        all_scores_pos = template[
            :, all_labels[0][0], all_labels[1][0], all_labels[2][0]
        ]
    else:
        all_scores_pos = np.ones((template.shape[0], 0))

    if len(all_labels[0][1]) > 0:
        all_scores_neg = template[
            :, all_labels[0][1], all_labels[1][1], all_labels[2][1]
        ]
    else:
        all_scores_neg = np.zeros((template.shape[0], 0))

    all_label_pos = np.ones(all_scores_pos.shape[1])
    all_label_neg = np.zeros(all_scores_neg.shape[1])

    if all_scores_pos.shape[1] == 0 and all_scores_neg.shape[1] == 0:
        x_filter, y_filter = None, None
    else:
        x_filter = (
            torch.from_numpy(np.hstack((all_scores_pos, all_scores_neg)))
            .float()
            .permute(1, 0)
        )
        y_filter = torch.from_numpy(
            np.concatenate((all_label_pos, all_label_neg))
        ).float()

    # Re-trained BLR model
    # Fit entire tomograms
    index_ = tm_idx
    x_onehot = torch.zeros(
        (x_filter.size(1), x_filter.size(1)),
        dtype=x_filter.dtype,
        device=x_filter.device,
    )

    y_onehot = torch.zeros(
        x_filter.size(1),
        dtype=y_filter.dtype,
        device=y_filter.device,
    )
    y_onehot[index_] = 1

    x_filter = torch.cat((x_filter, x_onehot), dim=0)
    y_filter = torch.cat((y_filter, y_onehot), dim=0)

    model.fit(
        x,
        y.ravel(),
        weights=count.ravel(),
        all_labels=[x_filter, y_filter],
    )

    weights = ', '.join(map(str, model.weights.numpy()))
    bias = ', '.join(map(str, model.bias.numpy()))
    weights_bias = weights + '|' + bias

    return weights_bias


@app.get("/new_proposal")
async def new_proposal(
    f_name: str,
    patch_corner: str,
    patch_size: int,
    pi: float,
    weights: str,
    bias: str,
):
    weights = weights.split(",")
    weights = tuple(map(float, weights))
    weights = torch.from_numpy(np.array(weights).astype(np.float32))

    bias = float(bias)
    bias = torch.Tensor([bias])

    model = BinaryLogisticRegression(
        n_features=len(weights),
        l2=1.0,
        pi=float(pi),
        pi_weight=1000,
    )
    model.fit(pre_train=[weights, bias])

    tomogram, _, _ = load_tomogram("data/" + f_name + f"/{f_name}.mrc", aws=True)

    pdb_name = template_list(f_name)
    template, _ = load_template(pdb_name)

    _, tm_score = draw_patch_and_scores(
        tomogram, template, patch_corner, patch_size
    )

    # BLR training and model update
    shape = tm_score.shape[1:]
    x = torch.from_numpy(tm_score.copy()).float()
    x = x.permute(1, 2, 3, 0)
    x = x.reshape(-1, x.shape[-1])

    with torch.no_grad():
        logits = model(x).reshape(*shape)
        logits_patch = torch.sigmoid(logits).cpu().detach().numpy()
        logits = logits.cpu().detach()

    # Draw 10 coordinates with lowest entropy
    proposals = rank_candidate_locations(logits, shape)
    patch_points = np.vstack(proposals[:10])

    logits_patch_shape = logits_patch.shape

    return logits_patch, logits_patch_shape, patch_points
