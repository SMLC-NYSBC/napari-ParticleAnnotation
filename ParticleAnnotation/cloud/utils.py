import io
from os import listdir

import torch
import numpy as np


def numpy_array_to_bytes_io(array: np.ndarray) -> io.BytesIO:
    if isinstance(array, torch.Tensor):
        if array.device.type == 'cpu':
            array = array.detach().numpy()
        else:
            array = array.cpu().detach().numpy()

    bytes_io = io.BytesIO()
    np.save(bytes_io, array, allow_pickle=False)
    bytes_io.seek(0)

    return bytes_io


def bytes_io_to_numpy_array(bytes_file) -> np.ndarray:
    bytes_file = io.BytesIO(bytes_file)

    return np.load(bytes_file, allow_pickle=True)


def get_model_name_and_weights(m_name, dir_):
    list_model = listdir(dir_ + "data/models")
    model_ids = [str(f[len(f) - 7 : -4]) for f in list_model if f.endswith("pth")]
    m_model_id = m_name[len(m_name) - 7 : -4]

    if m_model_id not in model_ids:
        new_id = [int(i) for i in model_ids]
        new_id = new_id[max(new_id)]
        m_name = f"topaz_al_model_{new_id:03}.pth"
        AL_weights = None
    else:
        weights = torch.load(dir_ + "data/models/" + m_name)
        AL_weights = [weights.weights, weights.bias]

    return m_name, AL_weights


def build_consensus(points: np.ndarray, multi=False) -> np.ndarray:
    """
    From multiple selection of the same point output unified consensus of a label

    Args:
        points (np.ndarray): Array of points to marge. Allow for single or multiple selections.
            If consensus should be built for multiple points, it requires ID label in the first column
        multi (bool): If True, expect point IDs in the first column

    Returns:
        np.ndarray: single consensus point as an array of shape [N, (2,3)]
    """
    if multi:
        unique_ids = np.unique(points[:, 0])
        consensus_list = [
            np.mean(points[np.where(points[:, 0] == u)[0], 1:], axis=1)
            for u in unique_ids
        ]
        consensus = np.concatenate(consensus_list)
    else:
        consensus = np.mean(points, axis=1)

    return consensus
