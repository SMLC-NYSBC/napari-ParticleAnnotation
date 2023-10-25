import torch.nn.functional as F
from scipy.optimize import minimize
import numpy as np
from topaz.model.factory import load_model
import torch

from ParticleAnnotation.utils.model.utils import find_peaks, get_device


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


def initialize_model(mrc, n_part=10):
    device_ = get_device()

    if len(mrc.shape) == 3:
        model = load_model("resnet8_u32")
        classifier = model.classifier.to(device_)
        model = model.features.to(device_)
        model.fill()
        model.eval()
        classifier.eval()

        mrc = torch.from_numpy(mrc).float().unsqueeze(0).to(device_)
        _, d, h, w = mrc.shape

        filter_values = torch.zeros((64, d, h, w)).to(device_)  # C D H W
        classified = np.zeros((64, d, h, w))  # C D H W

        from tqdm import tqdm

        for i in tqdm(range(mrc.shape[1])):
            with torch.no_grad():
                j = model(mrc[:, i, ...]).squeeze(0)
                filter_values[:, i, :] = j
                classified[:, i, :] = torch.sigmoid(classifier(j))

        x = filter_values.permute(1, 2, 3, 0)  # D, H, W, C
    else:
        model = load_model("resnet16")
        classifier = model.classifier.to(device_)
        model = model.features.to(device_)
        model.fill()
        model.eval()
        classifier.eval()

        with torch.no_grad():
            filter_values = model(torch.from_numpy(mrc).float().unsqueeze(0)).squeeze(0)
            classified = torch.sigmoid(classifier(filter_values))

        x = filter_values.permute(1, 2, 0)

    x = x.detach().cpu().numpy()
    classified = classified.detach().cpu().numpy()

    x = x.reshape(-1, x.shape[-1])  # L, C
    y = torch.zeros(len(x)) + np.nan

    # Classified particles
    xy, score = find_peaks(classified[0, :], with_score=True)
    xy_negative = xy[[np.array(score).argsort()[:n_part][::-1]], :][0, ...]  # Bottom 10

    xy_negative = np.hstack((np.zeros((xy_negative.shape[0], 1)), xy_negative))
    p_xy = xy_negative

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

        return loss

    def predict(self, x):
        logits = torch.matmul(x, self.weights) + self.bias
        return logits

    def __call__(self, x):
        return self.predict(x)

    def fit(self, x, y, weights=None, pre_train=None):
        if pre_train is not None:
            self.weights = pre_train[0]
            self.bias = pre_train[1]
        else:
            n_features = x.shape[1]
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

                loss = model.loss(x, y, weights=weights)
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
