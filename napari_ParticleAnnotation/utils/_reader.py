from typing import Union, Sequence, List, Callable, Optional
from napari_ParticleAnnotation.utils.load_data import load_image, load_xyz
from napari.types import LayerData

PathOrPaths = Union[str, Sequence[str]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def get_reader_img(path: PathOrPaths) -> Optional[ReaderFunction]:
    """
    Returns a function that can read image data from a file with the specified path.

    Args:
        path: A string or sequence of strings representing the path(s) of the file(s) to read.

    Returns:
        A function that can read image data from the specified file(s), or None if the file extension is not supported.
    """
    if isinstance(path, list):
        # If the input is a list, assume it represents an image stack and look at the first file only.
        path = path[0]

    # Check if the file extension is supported.
    extensions = (".mrc", ".mrcs", ".map")
    if not path.endswith(extensions):
        return None

    # Return the function that can read image data from the file.
    return load_image


def get_reader_xyz(path: PathOrPaths) -> Optional[ReaderFunction]:
    """
    Returns a function that can read XYZ data from a file with the specified path.

    Args:
        path: A string or sequence of strings representing the path(s) of the file(s) to read.

    Returns:
        A function that can read XYZ data from the specified file(s), or None if the file extension is not supported.
    """
    # Check if the file extension is supported.
    extensions = (".npy", ".csv", ".star")
    if not path.endswith(extensions):
        return None

    # Return the function that can read XYZ data from the file.
    return load_xyz
