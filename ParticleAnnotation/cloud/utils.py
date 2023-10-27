import io
import torch
import numpy as np


def numpy_array_to_bytes_io(array: np.ndarray) -> io.BytesIO:
    bytes_io = io.BytesIO()
    np.save(bytes_io, array, allow_pickle=False)
    bytes_io.seek(0)
    return bytes_io


def bytes_io_to_numpy_array(bytes_file) -> np.ndarray:
    bytes_file = io.BytesIO(bytes_file)

    return np.load(bytes_file, allow_pickle=True)


def get_model_name_and_weights(m_name, model_ids, dir_):
    def generate_new_model_name(ids):
        if ids:
            return max(ids) + 1
        return 0

    if m_name[len(m_name) - 7 : -4] not in model_ids:
        model_id = generate_new_model_name(model_ids)
        m_name = f"topaz_al_model_{model_id:03}.pth"
        state_name = f"state_{model_id:03}.pth"
        AL_weights = None
    else:
        weights = torch.load(dir_ + "data/models/" + m_name)
        AL_weights = [weights.weight, weights.bias]
        state_name = f"state_{m_name[len(m_name) - 7 : -4]:03}"

    return m_name, state_name, AL_weights


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
