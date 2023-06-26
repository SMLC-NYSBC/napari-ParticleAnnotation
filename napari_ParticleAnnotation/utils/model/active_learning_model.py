import torch.nn.functional as F
from scipy.optimize import minimize
import numpy as np
from topaz.model.factory import load_model
import torch

from napari_ParticleAnnotation.utils.model.utils import find_peaks


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
            for z, i, j, label in points:
                fill_label_region(y, i, j, label, int(size), z)
    else:
        if len(points) > 0:
            for i, j, label in points:
                fill_label_region(y, i, j, label, int(size))
    return y


def initialize_model(mrc, troch=None):
    model = load_model("resnet16")
    classifier = model.classifier
    model = model.features
    model.fill()
    model.eval()
    classifier.eval()

    if len(mrc.shape) == 3:
        mrc = torch.from_numpy(mrc).float().unsqueeze(0)
        _, d, h, w = mrc.shape

        filter_values = torch.zeros((256, d, h, w))  # C D H W
        from tqdm import tqdm

        for i in tqdm(range(mrc.shape[1])):
            with torch.no_grad():
                j = model(mrc[:, i, ...]).squeeze(0)
                filter_values[:, i, :] = j

        x = filter_values.numpy().transpose([1, 2, 3, 0])  # D, H, W, C
    else:
        with torch.no_grad():
            filter_values = model(torch.from_numpy(mrc).float().unsqueeze(0)).squeeze(0)
            classified = torch.sigmoid(classifier(filter_values)).numpy()
            filter_values = filter_values.numpy()  # C, W, H

        x = filter_values.transpose([1, 2, 0])  # W, H, C

    x = x.reshape(-1, x.shape[-1])  # L, C
    x = torch.from_numpy(x).float()
    y = torch.zeros(len(x)) + np.nan

    # Classified particles
    xy, score = find_peaks(classified[0, :], with_score=True)
    xy_negative = xy[np.where(np.array(score) < 0.01)[0], :]
    xy_positive = xy[np.where(np.array(score) > 0.8)[0], :]

    xy_negative = np.hstack((np.zeros((xy_negative.shape[0], 1)), xy_negative))
    xy_positive = np.hstack((np.ones((xy_positive.shape[0], 1)), xy_positive))
    p_xy = np.vstack((xy_negative, xy_positive))

    return x, y, p_xy


class BinaryLogisticRegression:
    def __init__(self, n_features, l2=1.0, pi=0.01, pi_weight=1.0) -> None:
        self.weights = torch.zeros(n_features)
        self.bias = torch.zeros(1)
        self.l2 = l2
        self.pi = pi
        self.pi_logit = np.log(pi) - np.log1p(-pi)
        self.pi_weight = pi_weight

    def loss(self, x, y, weights=None):
        logits = torch.matmul(x, self.weights) + self.bias

        if weights is None:
            weights = torch.ones_like(y)

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

        return loss, (loss_binary / n, loss_reg_l2, loss_pi)

    def predict(self, x):
        logits = torch.matmul(x, self.weights) + self.bias
        return logits

    def __call__(self, x):
        return self.predict(x)

    def fit(self, x, y, weights=None):
        n_features = x.shape[1]
        # theta0 = torch.concat([self.weights, self.bias]).detach().numpy()
        theta0 = np.zeros(n_features + 1)

        def loss_fn(theta):
            w = torch.from_numpy(theta[:n_features]).float()
            b = torch.from_numpy(theta[n_features:]).float()
            w.requires_grad = True
            b.requires_grad = True

            model = BinaryLogisticRegression(
                n_features, l2=self.l2, pi=self.pi, pi_weight=self.pi_weight
            )
            model.weights = w
            model.bias = b

            loss, _ = model.loss(x, y, weights=weights)
            loss.backward()

            grad = torch.concat([w.grad, b.grad]).detach().numpy()
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

