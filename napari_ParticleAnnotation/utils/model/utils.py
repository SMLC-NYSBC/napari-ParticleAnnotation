import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter


size = (37, 37)


def find_peaks(score, size=size[0] / 3):
    max_filter = maximum_filter(score, size=size)
    peaks = score - max_filter
    peaks = np.where(peaks == 0)
    peaks = np.stack(peaks, axis=-1)
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
    peak_scores = entropy[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(peak_scores)
    ordered = peaks[order]

    cur_proposal_index, proposals = set_proposals(ordered, proposals, id_)

    return cur_proposal_index, proposals
