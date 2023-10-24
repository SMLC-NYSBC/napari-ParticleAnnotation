import io
import numpy as np


def numpy_array_to_bytes_io(array: np.ndarray) -> io.BytesIO:
    bytes_io = io.BytesIO()
    np.save(bytes_io, array, allow_pickle=False)
    bytes_io.seek(0)
    return bytes_io


def bytes_io_to_numpy_array(bytes_io: io.BytesIO) -> np.ndarray:
    bytes_io.seek(0)
    return np.load(bytes_io, allow_pickle=False)
