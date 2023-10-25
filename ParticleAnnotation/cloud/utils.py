import io
import numpy as np
from topaz import torch


def numpy_array_to_bytes_io(array: np.ndarray) -> io.BytesIO:
    bytes_io = io.BytesIO()
    np.save(bytes_io, array, allow_pickle=False)
    bytes_io.seek(0)
    return bytes_io


def bytes_io_to_numpy_array(bytes_file) -> np.ndarray:
    bytes_file = io.BytesIO(bytes_file)

    return np.load(bytes_file, allow_pickle=True)


def get_model_name_and_weights(m_name, model_ids, dir_):
    def load_weights(name):
        weights = torch.load(dir_ + "data/models/" + name)
        return [weights.weight, weights.bias]

    def generate_new_model_name(ids):
        if ids:
            return max(ids) + 1
        return 0

    if m_name is None or m_name[len(m_name) - 7:-4] not in model_ids:
        model_id = generate_new_model_name(model_ids)
        m_name = f"topaz_al_model_{model_id:03}.pth"
        state_name = f"state_{model_id:03}.pth"
        AL_weights = None
    else:
        AL_weights = load_weights(m_name)

    return m_name, state_name, AL_weights
