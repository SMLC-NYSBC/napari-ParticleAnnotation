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
        data[:, 0] = data[:, 0] + patch_corner[0]
        data[:, 1] = data[:, 1] + patch_corner[1]
        data[:, 2] = data[:, 2] + patch_corner[2]
    else:
        data[:, 0] = data[:, 0] - patch_corner[0]
        data[:, 1] = data[:, 1] - patch_corner[1]
        data[:, 2] = data[:, 2] - patch_corner[2]

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


def get_random_patch(img, size_: int, chosen_particles=None):
    z, y, x = img.shape
    print(f"image shape is - {img.shape}")
    if chosen_particles is None or chosen_particles.shape[0] == 0:
        if img.shape[0] > size_:
            z_start = np.random.randint(0, z - size_ + 1)
        if img.shape[1] > size_:
            y_start = np.random.randint(0, y - size_ + 1)
        if img.shape[2] > size_:
            x_start = np.random.randint(0, x - size_ + 1)
        chosen_particles = None
    else:
        center_idx = np.random.randint(0, chosen_particles.shape[0])
        center = chosen_particles[center_idx]
        print(f"center is - {center}")
        center_z, center_y, center_x = center

        if img.shape[0] > size_:
            z_start = max(0, min(center_z - size_ // 2, z - size_))
            z_start = int(z_start)

        if img.shape[1] > size_:
            y_start = max(0, min(center_y - size_ // 2, y - size_))
            y_start = int(y_start)

        if img.shape[2] > size_:
            x_start = max(0, min(center_x - size_ // 2, x - size_))
            x_start = int(x_start)

        # [TO-DO] add non-max suppression here to choose the best centers
        idx = np.where((chosen_particles == center).all(axis=1))
        chosen_particles = np.delete(chosen_particles, idx, axis=0)

    if img.shape[0] <= size_:
        z_start = 0
        z_end = z
    else:
        z_end = z_start + size_

    if img.shape[1] <= size_:
        y_start = 0
        y_end = y
    else:
        y_end = y_start + size_

    if img.shape[2] <= size_:
        x_start = 0
        x_end = x
    else:
        x_end = x_start + size_

    # Extract the patch from the array
    patch = img[z_start:z_end, y_start:y_end, x_start:x_end]

    return patch, (z_start, y_start, x_start), chosen_particles


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


def find_peaks(score, size=size[0] / 3, with_score=False):
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()

    max_filter = maximum_filter(score, size=size)
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
        return peaks, scores
    return peaks


def set_proposals(ordered, proposals, id_=1):
    proposals.clear()
    proposals.extend(ordered)
    cur_proposal_index = len(proposals) - id_

    return cur_proposal_index, proposals


def rank_candidate_locations(model, x, shape, proposals, id_=1):
    # rank the candidates by entropy - ask the user to label
    # the highest entropy location, which is where the model has the most uncertainty
    # about the label
    # this is not an optimal strategy, but it works fine for this prototype
    with torch.no_grad():
        logits = model(x)
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

    cur_proposal_index, proposals = set_proposals(ordered, proposals, id_)

    return cur_proposal_index, proposals


def get_device() -> torch.device:
    """
    Return device that can be used for training or predictions

    Returns:
        torch.device: Device type.
    """
    df = torch.rand((1, 1))
    device = torch.device("cpu")
    # try:
    #     device = torch.device("cuda:0")
    #     df.to(device)
    # except AssertionError:
    #     device = torch.device("cpu")
    #     df.to(device)
    return device
