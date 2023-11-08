import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, convolve


size = (37, 37)


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
    try:
        device = torch.device("cuda:0")
        df.to(device)
    except AssertionError:
        device = torch.device("cpu")
        df.to(device)
    return device
