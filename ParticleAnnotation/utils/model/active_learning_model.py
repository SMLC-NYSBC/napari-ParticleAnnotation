import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
import numpy as np
from topaz.model.factory import load_model
import torch
from tqdm import tqdm

from ParticleAnnotation.utils.model.utils import (
    find_peaks,
    get_device,
    divide_grid,
    correct_coord,
)
import io
import requests


def predict_3d_with_AL(img, model, weights, offset, tm_scores=None):
    peaks, peaks_logits = [], []
    device_ = get_device()

    grid = divide_grid(img, offset)
    if tm_scores is None:
        init_model = torch.load(
            io.BytesIO(
                requests.get(
                    "https://topaz-al.s3.dualstack.us-east-1.amazonaws.com/topaz3d.sav",
                    timeout=(5, None),
                ).content
            )
        )
        init_model = init_model.features.to(device_)
        init_model.fill()
        init_model.eval()

    if weights is not None:
        model.fit(pre_train=weights)

    for i in tqdm(grid):
        # Stream patch
        patch = img[i[0] : i[0] + offset, i[1] : i[1] + offset, i[2] : i[2] + offset]
        shape_ = patch.shape

        # Predict
        with torch.no_grad():
            patch = torch.from_numpy(patch).float().unsqueeze(0).to(device_)
            if tm_scores is None:
                patch = init_model(patch).squeeze(0).permute(1, 2, 3, 0)
            else:
                z_start, y_start, x_start = i[0], i[1], i[2]
                patch = tm_scores[
                    :,
                    z_start : z_start + offset,
                    y_start : y_start + offset,
                    x_start : x_start + offset,
                ]
                patch = torch.from_numpy(patch).float().to(device_)
                patch = patch.permute(1, 2, 3, 0)
                patch = patch.reshape(-1, patch.shape[-1])
            logits = model(patch).reshape(*shape_)

        if device_ == "cpu":
            logits = logits.detach().numpy()
        else:
            logits = logits.cpu().detach().numpy()

        # Extract peaks
        max_filter = maximum_filter(logits, size=25)
        peaks_df = logits - max_filter
        peaks_df = np.where(peaks_df == 0)
        peaks_df = np.stack(peaks_df, axis=-1)

        # Save patch peaks and its logits
        peaks_logits_df = logits[peaks_df[:, 0], peaks_df[:, 1], peaks_df[:, 2]]
        peaks_df = correct_coord(peaks_df, i, True)
        peaks.append(peaks_df)
        peaks_logits.append(peaks_logits_df)

    print("Done with AL training")

    return np.vstack(peaks), np.concatenate(peaks_logits)


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
    else:
        mask = neg_mask

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


def label_points_to_mask(points, shape, size):
    y = torch.zeros(*shape) + np.nan

    if len(shape) == 3:
        if len(points) > 0:
            for label, z, i, j in points:
                fill_label_region(y, i, j, label, int(size), z)
    else:
        if len(points) > 0:
            for label, i, j in points:
                fill_label_region(y, i, j, label, int(size))
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
    def __init__(self, n_features, l2=1.0, pi=0.01, pi_weight=1.0) -> None:
        self.device = get_device()
        # self.weights = torch.zeros(n_features, device=self.device)
        # random initialization
        self.weights = torch.randn(n_features, device=self.device)
        # self.bias = torch.zeros(1, device=self.device)
        self.bias = torch.randn(1, device=self.device)
        self.l2 = l2
        self.pi = pi
        self.pi_logit = np.log(pi) - np.log1p(-pi)
        self.pi_weight = pi_weight

    def loss(self, x, y, weights=None):
        logits = torch.matmul(x, self.weights) + self.bias

        if weights is None:
            weights = torch.ones_like(y, device=self.device)

        # binary cross entropy for labeled y's
        is_labeled = ~torch.isnan(y)
        weights = weights[is_labeled]
        n = torch.sum(weights)
        loss_binary = F.binary_cross_entropy_with_logits(
            logits[is_labeled], y[is_labeled], reduction="sum", weight=weights
        )
        # L2 regularizer on the weights
        loss_reg_l2 = self.l2 * torch.sum(self.weights**2) / 2

        # Penalty on the expected pi
        log_p = torch.logsumexp(F.logsigmoid(logits), dim=0) - np.log(len(logits))
        log_np = torch.logsumexp(F.logsigmoid(-logits), dim=0) - np.log(len(logits))
        logit_expect = log_p - log_np
        loss_pi = self.pi_weight * (logit_expect - self.pi_logit) ** 2

        loss = (loss_binary + loss_reg_l2 + loss_pi) / n

        return loss

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        else:
            x = x.to(self.device)
        return torch.matmul(x, self.weights) + self.bias

    def __call__(self, x):
        return self.predict(x)

    def fit(self, x=None, y=None, weights=None, pre_train=None):
        if pre_train is not None:
            self.weights = pre_train[0]
            self.weights = self.weights.to(self.device)
            self.bias = pre_train[1]
            self.bias = self.bias.to(self.device)
        else:
            x, y = x.to(self.device), y.to(self.device)

            if weights is not None:
                weights = weights.to(self.device)

            n_features = x.shape[1]
            theta0 = np.zeros(n_features + 1)

            def loss_fn(theta):
                w = torch.from_numpy(theta[:n_features]).float().to(self.device)
                b = torch.from_numpy(theta[n_features:]).float().to(self.device)
                w.requires_grad = True
                b.requires_grad = True

                model = BinaryLogisticRegression(
                    n_features, l2=self.l2, pi=self.pi, pi_weight=self.pi_weight
                )
                model.weights = w
                model.bias = b

                loss = model.loss(x, y, weights=weights)
                loss.backward()

                grad = torch.concat([w.grad, b.grad]).cpu().detach().numpy()
                loss = loss.item()

                return loss, grad

            result = minimize(loss_fn, theta0, jac=True)
            self.result = result

            theta = result.x
            w = torch.from_numpy(theta[:n_features]).float()
            b = torch.from_numpy(theta[n_features:]).float()
            self.weights = w
            self.bias = b

        return self
