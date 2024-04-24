import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
import numpy as np
from topaz.model.factory import load_model
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from ParticleAnnotation.utils.model.utils import (
    find_peaks,
    get_device,
    divide_grid,
)
import io
import requests

from ParticleAnnotation.utils.pc_sampling import VoxelDownSampling


def predict_3d_with_AL(
    img: np.ndarray,
    tm_scores: np.ndarray,
    model,
    offset: int,
    maximum_filter_size=25,
    gauss_filter=0.0,
    filament=False,
):
    peaks, peaks_logits = [], []
    device_ = get_device()

    assert tm_scores.shape[1:] == img.shape

    # if tm_scores is None:
    #     init_model = torch.load(
    #         io.BytesIO(
    #             requests.get(
    #                 "https://topaz-al.s3.dualstack.us-east-1.amazonaws.com/topaz3d.sav",
    #                 timeout=(5, None),
    #             ).content
    #         )
    #     )
    #     init_model = init_model.features.to(device_)
    #     init_model.fill()
    #     init_model.eval()

    # if weights is not None:
    #     model.fit(pre_train=weights)

    grid = divide_grid(img, offset)
    full_logits = np.zeros_like(img)
    for i in tqdm(grid):
        # Predict
        with torch.no_grad():
            # if tm_scores is None:
            #     # Stream patch
            #     patch = img[
            #         i[0] : i[0] + offset, i[1] : i[1] + offset, i[2] : i[2] + offset
            #     ]
            #     shape_ = patch.shape
            #     patch = torch.from_numpy(patch).float().unsqueeze(0).to(device_)

            #     patch = init_model(patch).squeeze(0).permute(1, 2, 3, 0)
            # else:
            z_start, y_start, x_start = i[0], i[1], i[2]
            patch = tm_scores[
                :,
                z_start : z_start + offset,
                y_start : y_start + offset,
                x_start : x_start + offset,
            ]
            shape_ = patch.shape

            patch = torch.from_numpy(patch.copy()).float().to(device_)
            patch = patch.permute(1, 2, 3, 0)
            patch = patch.reshape(-1, patch.shape[-1])

            logits = model(patch).reshape(*shape_[1:])

            if device_ == "cpu":
                logits = logits.detach().numpy()
            else:
                logits = logits.cpu().detach().numpy()

            full_logits[
                z_start : z_start + offset,
                y_start : y_start + offset,
                x_start : x_start + offset,
            ] = logits

    full_logits = torch.sigmoid(torch.from_numpy(full_logits)).detach().numpy()
    if gauss_filter > 0:
        full_logits = gaussian_filter(full_logits, sigma=gauss_filter)

    # Extract peaks
    if filament:
        from skimage.morphology import skeletonize_3d

        peaks = skeletonize_3d(np.where(full_logits > 0.5, 1, 0))
        peaks = np.where(peaks > 0)
        peaks = np.stack((peaks[0], peaks[1], peaks[2])).T
        peaks = VoxelDownSampling(voxel=5, labels=False, KNN=False)(coord=peaks)
        peaks = peaks.astype(np.int16)
    else:
        max_filter = maximum_filter(
            full_logits.astype(np.float32), size=maximum_filter_size
        )
        peaks = full_logits - max_filter
        peaks = np.where(peaks == 0)
        peaks = np.stack(peaks, axis=-1)

    # Save patch peaks and its logits
    peaks_logits = full_logits[peaks[:, 0], peaks[:, 1], peaks[:, 2]]

    return peaks, peaks_logits, full_logits


def fill_label_region(y, ci, cj, label, size: int, cz=None):
    neg_radius = size
    pos_radius = size

    # Centralizer mask
    r = max(neg_radius, pos_radius)
    k = r * 2
    if k % 2 == 0:
        if pos_radius % 2 == 0:
            k = k + 1
    else:
        if pos_radius % 2 == 0:
            k = k + 1

    if cz is not None:
        center = (k // 2, k // 2, k // 2)
        grid = np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij")
    else:
        center = (k // 2, k // 2)
        grid = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    grid = np.stack(grid, axis=-1)

    d = np.sqrt(np.sum((grid - center) ** 2, axis=-1))

    pos_mask = np.zeros_like(d, dtype=bool)
    neg_mask = np.zeros_like(d, dtype=bool)
    if cz is not None:
        start = center - np.repeat(pos_radius // 2, 3)
        end = start + pos_radius

        pos_mask[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = True
        neg_mask[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = True
    else:
        start = center - np.repeat(pos_radius // 2, 2)
        end = start + pos_radius
        pos_mask[start[0] : end[0], start[1] : end[1]] = True
        neg_mask[start[0] : end[0], start[1] : end[1]] = True

    if label == 1:
        mask = pos_mask
    elif label == 0:
        mask = neg_mask
    else:
        return y

    if cz is not None:
        k = mask.shape[0]

        dz, di, dj = np.where(mask)
        z = cz + dz - k // 2
        i = ci + di - k // 2
        j = cj + dj - k // 2

        keep = (
            (0 <= z)
            & (z < y.shape[0])
            & (0 <= i)
            & (i < y.shape[1])
            & (0 <= j)
            & (j < y.shape[2])
        )
        z = z[keep]
        i = i[keep]
        j = j[keep]

        y[z, i, j] = label
    else:
        k = mask.shape[0]

        di, dj = np.where(mask)
        i = ci + di - k // 2
        j = cj + dj - k // 2

        keep = (0 <= i) & (i < y.shape[0]) & (0 <= j) & (j < y.shape[1])
        i = i[keep]
        j = j[keep]

        y[i, j] = label
    return y


def stack_all_labels(scores, points, size):
    label_radius = size
    r = label_radius
    k = r * 2

    if k % 2 == 0:
        if label_radius % 2 == 0:
            k = k + 1
    else:
        if label_radius % 2 == 0:
            k = k + 1

    center = (k // 2, k // 2, k // 2)
    grid = np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij")

    grid = np.stack(grid, axis=-1)
    d = np.sqrt(np.sum((grid - center) ** 2, axis=-1))

    start = center - np.repeat(label_radius // 2, 3)
    end = start + label_radius

    mask = np.zeros_like(d, dtype=bool)
    mask[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = True

    k = mask.shape[0]

    z_list_pos, z_list_neg = [[]], [[]]
    i_list_pos, i_list_neg = [[]], [[]]
    j_list_pos, j_list_neg = [[]], [[]]
    for x in points:
        label, cz, ci, cj = x
        if label not in [0, 1]:
            continue

        dz, di, dj = np.where(mask)
        z = cz + dz - k // 2
        i = ci + di - k // 2
        j = cj + dj - k // 2

        keep = (
            (0 <= z)
            & (z < scores.shape[1])
            & (0 <= i)
            & (i < scores.shape[2])
            & (0 <= j)
            & (j < scores.shape[3])
        )

        if label == 1:
            z_list_pos.append(z[keep])
            i_list_pos.append(i[keep])
            j_list_pos.append(j[keep])
        else:
            z_list_neg.append(z[keep])
            i_list_neg.append(i[keep])
            j_list_neg.append(j[keep])

    z_list_pos = np.concatenate(z_list_pos).astype(np.int16)
    i_list_pos = np.concatenate(i_list_pos).astype(np.int16)
    j_list_pos = np.concatenate(j_list_pos).astype(np.int16)

    z_list_neg = np.concatenate(z_list_neg).astype(np.int16)
    i_list_neg = np.concatenate(i_list_neg).astype(np.int16)
    j_list_neg = np.concatenate(j_list_neg).astype(np.int16)

    return (z_list_pos, z_list_neg), (i_list_pos, i_list_neg), (j_list_pos, j_list_neg)


def label_points_to_mask(points, shape, size):
    y = torch.zeros(*shape) + torch.nan

    if len(shape) == 3:
        if len(points) > 0:
            for label, z, i, j in points:
                y = fill_label_region(y, i, j, label, int(size), z)
    else:
        if len(points) > 0:
            for label, i, j in points:
                y = fill_label_region(y, i, j, label, int(size))
    return y


def update_true_labels(true_labels, points_layer, label):
    data = np.asarray(points_layer)
    if data.shape[1] == 2:
        data = np.array((np.array(label).astype(np.int16), data[:, 0], data[:, 1])).T
    else:
        data = np.array(
            (
                np.array(label).astype(np.int16),
                data[:, 0],
                data[:, 1],
                data[:, 2],
            )
        ).T

    for data_ in data:
        if data_[0] == 1:
            true_labels = np.vstack((true_labels, data_)) if true_labels.size else data_

    true_labels = np.unique(true_labels, axis=0)

    return true_labels


def initialize_model(mrc, n_part=10, only_feature=False, tm_scores=None):
    device_ = get_device()

    if len(mrc.shape) == 3:
        if tm_scores is not None:
            if not isinstance(tm_scores, torch.Tensor):
                x = torch.from_numpy(tm_scores.copy()).float()
            else:
                x = tm_scores

            classified = x
            print("Chosen to use TM scores as features")
        else:
            model = torch.load(
                io.BytesIO(
                    requests.get(
                        "https://topaz-al.s3.dualstack.us-east-1.amazonaws.com/topaz3d.sav",
                        timeout=(5, None),
                    ).content
                )
            )
            classifier = model.classifier.to(device_)
            model = model.features.to(device_)
            model.fill()
            model.eval()
            classifier.eval()
            # see classifier
            mrc = torch.from_numpy(mrc).float().unsqueeze(0).to(device_)

            with torch.no_grad():
                filter_values = model(mrc).squeeze(0)
                classified = torch.sigmoid(classifier(filter_values))

            x = filter_values.permute(1, 2, 3, 0)
            print("Chosen to use the Topaz features")
    else:
        model = load_model("resnet16")
        classifier = model.classifier.to(device_)
        model = model.features.to(device_)
        model.fill()
        model.eval()
        classifier.eval()
        mrc = torch.from_numpy(mrc).float().unsqueeze(0).to(device_)

        with torch.no_grad():
            filter_values = model(mrc).squeeze(0)
            classified = torch.sigmoid(classifier(filter_values))

        x = filter_values.permute(1, 2, 0)

    x = x.reshape(-1, x.shape[-1])  # L, C
    if only_feature:
        return x

    if isinstance(classified, torch.Tensor):
        classified = classified.detach().cpu().numpy()

    y = torch.zeros(len(x)) + np.nan
    # Classified particles
    xy, score = find_peaks(classified[0, :], with_score=True)

    xy_negative = xy[[np.array(score).argsort()[:n_part][::-1]], :][0, ...]
    xy_positive = xy[[np.array(score).argsort()[-n_part:][::-1]], :][
        0, ...
    ]  # choose top 1000

    xy_negative = np.hstack((np.zeros((xy_negative.shape[0], 1)), xy_negative))
    xy_positive = np.hstack((np.ones((xy_positive.shape[0], 1)), xy_positive))
    p_xy = (xy_negative, xy_positive)

    return x, y, p_xy


class BinaryLogisticRegression:
    def __init__(self, n_features, l2=1.0, pi=0.01, pi_weight=1.0, ice=True) -> None:
        """
        Make a dataset where for each feature [1 data poitn per n_features] its 0 or 1,
        1 is. Add scaler for the feateur, start with 1 scale to 128???

        ! Not Using pi! check if its not breaking anything else
        """
        self.n_features = n_features
        self.device = get_device()

        self.weights = torch.zeros(self.n_features, device=self.device)

        if ice:
            self.weights[-3:] = -1

        self.bias = torch.zeros(1, device=self.device)
        self.l2 = l2
        self.pi = pi
        self.pi_logit = np.log(pi) - np.log1p(-pi)
        self.pi_weight = pi_weight

    def loss(self, x, y, weights=None, all_x=None, all_y=None):
        logits = torch.matmul(x, self.weights) + self.bias

        if all_x is None and all_y is None:
            # logits = torch.matmul(x, self.weights) + self.bias
            if weights is None:
                weights = torch.ones_like(y, device=self.device)

            # binary cross entropy for labeled y's
            is_labeled = ~torch.isnan(y)
            weights = weights[is_labeled]
            n = torch.sum(weights)

            loss_binary = F.binary_cross_entropy_with_logits(
                logits[is_labeled], y[is_labeled], reduction="sum", weight=weights
            )
        else:
            logits_all = torch.matmul(all_x, self.weights) + self.bias

            weights = torch.ones_like(all_y, device=self.device)

            is_labeled = ~torch.isnan(all_y)
            weights = weights[is_labeled]
            logits_all = logits_all[is_labeled]
            all_y = all_y[is_labeled]

            n = torch.sum(weights)
            loss_binary = F.binary_cross_entropy_with_logits(
                logits_all, all_y, reduction="sum", weight=weights
            )

        # L2 regularizer on the weights
        loss_reg_l2 = self.l2 * torch.sum(self.weights**2) / 2

        # # Penalty on the expected pi
        log_p = torch.logsumexp(F.logsigmoid(logits), dim=0) - np.log(len(logits))
        log_np = torch.logsumexp(F.logsigmoid(-logits), dim=0) - np.log(len(logits))
        logit_expect = log_p - log_np
        loss_pi = self.pi_weight * (logit_expect - self.pi_logit) ** 2

        loss = (loss_binary + loss_reg_l2 + loss_pi) / n
        # loss = (loss_binary + loss_reg_l2) / n

        return loss

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        else:
            x = x.to(self.device)

        # x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        x = torch.matmul(x, self.weights) + self.bias

        return x

    def __call__(self, x):
        return self.predict(x)

    def fit(
        self, x=None, y=None, weights=None, pre_train=None, all_labels: list = None
    ):
        if pre_train is not None:
            self.weights = pre_train[0]
            self.weights = self.weights.to(self.device)
            self.bias = pre_train[1]
            self.bias = self.bias.to(self.device)
        else:
            x, y = x.to(self.device), y.to(self.device)
            if all_labels is not None:
                x_all, y_all = all_labels[0].to(self.device), all_labels[1].to(
                    self.device
                )
            else:
                x_all, y_all = None, None

            if weights is not None:
                weights = weights.to(self.device)

            theta0 = np.zeros(self.n_features + 1)

            def loss_fn(theta):
                w = torch.from_numpy(theta[: self.n_features]).float().to(self.device)
                b = torch.from_numpy(theta[self.n_features :]).float().to(self.device)
                w.requires_grad = True
                b.requires_grad = True

                model = BinaryLogisticRegression(
                    self.n_features, l2=self.l2, pi=self.pi, pi_weight=self.pi_weight
                )
                model.weights = w
                model.bias = b

                loss = model.loss(x, y, weights=weights, all_x=x_all, all_y=y_all)
                loss.backward()

                grad = torch.concat([w.grad, b.grad]).cpu().detach().numpy()
                loss = loss.item()

                return loss, grad

            result = minimize(loss_fn, theta0, jac=True)

            theta = result.x
            w = torch.from_numpy(theta[: self.n_features]).float()
            b = torch.from_numpy(theta[self.n_features :]).float()
            self.weights = w
            self.bias = b

        return self
