import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, convolve

size = (37, 37)


def divide_grid(array, size):
    # Get the shape of the array
    z, y, x = array.shape

    # Calculate the number of divisions along each axis
    nx, ny, nz = x // size, y // size, z // size

    nx = nx + 1 if x % size != 0 else nx
    ny = ny + 1 if y % size != 0 else ny
    nz = nz + 1 if z % size != 0 else nz

    # Initialize a list to hold the coordinates
    coordinates = []

    # Loop through each division and store the coordinates
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                left_corner = (i * size, j * size, k * size)
                coordinates.append(left_corner)

    return coordinates


def correct_coord(data, patch_corner, normalize):
    if normalize:
        data = data + patch_corner
    else:
        data = data - patch_corner
    return data


def calc_iou(box_1, box_2, size_):
    box_11 = (
        box_1[0] - size_ // 2,
        box_1[1] - size_ // 2,
        box_1[2] - size_ // 2,
        box_1[0] + size_ // 2,
        box_1[1] + size_ // 2,
        box_1[2] + size_ // 2,
    )
    box_22 = (
        box_2[0] - size_ // 2,
        box_2[1] - size_ // 2,
        box_2[2] - size_ // 2,
        box_2[0] + size_ // 2,
        box_2[1] + size_ // 2,
        box_2[2] + size_ // 2,
    )
    x1 = max(box_11[0], box_22[0])
    y1 = max(box_11[1], box_22[1])
    z1 = max(box_11[2], box_22[2])
    x2 = min(box_11[3], box_22[3])
    y2 = min(box_11[4], box_22[4])
    z2 = min(box_11[5], box_22[5])

    intersection = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)
    box1_vol = size_**3
    box2_vol = size_**3
    iou = intersection / (box1_vol + box2_vol - intersection)

    return iou


def get_random_patch(img_size, size_: int, chosen_particles=None):
    z, y, x = img_size

    if chosen_particles is None or chosen_particles.shape[0] == 0:
        if z > size_:
            z_start = np.random.randint(0, z - size_ + 1)
        else:
            z_start = 0
        if y > size_:
            y_start = np.random.randint(0, y - size_ + 1)
        else:
            y_start = 0
        if x > size_:
            x_start = np.random.randint(0, x - size_ + 1)
        else:
            x_start = 0
    else:
        center_idx = np.random.randint(0, chosen_particles.shape[0])
        center = chosen_particles[center_idx]
        center_z, center_y, center_x = center

        if z > size_:
            z_start = max(0, center_z - size_ // 2)
            z_start = int(z_start)
        else:
            z_start = 0

        if y > size_:
            y_start = max(0, center_y - size_ // 2)
            y_start = int(y_start)
        else:
            y_start = 0

        if x > size_:
            x_start = max(0, center_x - size_ // 2)
            x_start = int(x_start)
        else:
            x_start = 0

    return (z_start, y_start, x_start)


def sobel_filter(img):
    # Define Sobel operator kernels.
    kernel_x = np.array(
        [
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1],
        ]
    )

    kernel_y = np.array(
        [
            [1, 4, 6, 4, 1],
            [2, 8, 12, 8, 2],
            [0, 0, 0, 0, 0],
            [-2, -8, -12, -8, -2],
            [-1, -4, -6, -4, -1],
        ]
    )

    # Convolve image with kernels to get x and y derivatives of image.
    g_x = convolve(img, kernel_x)
    g_y = convolve(img, kernel_y)

    # Calculate magnitude of gradient as sqrt(g_x^2 + g_y^2).
    g = np.hypot(g_x, g_y)
    g *= 255.0 / np.max(g)  # normalize (scale) to 0-255

    return g


def polar_to_cartesian(rho, theta):
    x = rho * np.cos(np.radians(theta))
    y = rho * np.sin(np.radians(theta))
    return x, y


def find_peaks(score, size=15, with_score=False):
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()

    max_filter = maximum_filter(score.astype(np.float32), size=size)
    peaks = score - max_filter
    peaks = np.where(peaks == 0)
    peaks = np.stack(peaks, axis=-1)

    if with_score:
        scores = []
        if peaks.shape[1] == 3:
            for i in peaks:
                scores.append(score[i[0], i[1], i[2]])
        else:
            for i in peaks:
                scores.append(score[i[0], i[1]])

        order = np.argsort(scores)
        peaks = peaks[order]

        return np.vstack(peaks), np.vstack(scores)
    return peaks


def set_proposals(ordered, proposals, id_=1):
    proposals.clear()
    proposals.extend(ordered)
    cur_proposal_index = len(proposals) - id_

    return cur_proposal_index, proposals


def rank_candidate_locations(logits, shape):
    # rank the candidates by entropy - ask the user to label
    # the highest entropy location, which is where the model has the most uncertainty
    # about the label
    # this is not an optimal strategy, but it works fine for this prototype
    log_p = F.logsigmoid(logits).numpy()
    log_np = F.logsigmoid(-logits).numpy()
    entropy = -np.exp(log_p) * log_p - np.exp(log_np) * log_np

    # use peak finding to void finding candidates too close together (good for skip)
    entropy = entropy.reshape(*shape)
    peaks = find_peaks(entropy)

    if peaks.shape[1] == 3:
        peak_scores = entropy[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    else:
        peak_scores = entropy[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(peak_scores)
    ordered = peaks[order]

    return ordered


def get_device(device: str = "0") -> torch.device:
    """
    Return a device that can be used for training or predictions

    Args:
        device (str, int): Device name or ID.

    Returns:
        torch.device: Device type.
    """
    if device == "gpu":  # Load GPU ID 0
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif device == "cpu":  # Load CPU
        device = torch.device("cpu")
    elif device_is_str(device):  # Load specific GPU ID
        if torch.cuda.is_available():
            if int(device) == -1:
                device = torch.device("cpu")
            else:
                device = torch.device(f"cuda:{int(device)}")
        else:
            device = torch.device("cpu")
    elif device == "mps":  # Load Apple silicon
        if torch.backends.mps.is_available():
            device = torch.device("cpu")  # So far pytorch don't support CNN on MPS
        else:
            device = torch.device("cpu")
    # return device
    return torch.device("cpu")


def device_is_str(device: str = "0") -> bool:
    """
    Check if used device is convertible to int value

    Args:
        device (str, int): Device ID.

    Returns:
        bool: Check for input string/int
    """

    try:
        int(device)
        if isinstance(int(device), int):
            return True
        else:
            return False
    except ValueError:
        return False
