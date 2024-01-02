from typing import Union, Sequence
import numpy as np

PathOrPaths = Union[str, Sequence[str]]


def write_to_file(filename: str, file):
    coords = file[0][0]
    metadata = file[0][1]
    ids = metadata["face_color"]

    _, unique_ids = np.unique(ids, axis=0, return_inverse=True)
    data = np.hstack(
        (unique_ids.reshape(-1, 1).astype(np.int16), coords.astype(np.int16))
    )

    np.savetxt(fname=filename, X=data, delimiter=",")

    return filename
