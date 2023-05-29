import torch.nn.functional as F
from scipy.optimize import minimize
import numpy as np
from topaz.model.factory import load_model
import torch


def fill_label_region(y, ci, cj, label):
    neg_radius = 3
    pos_radius = 3

    r = max(neg_radius, pos_radius)
    k = r * 2 + 3
    center = (k // 2, k // 2)

    grid = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    grid = np.stack(grid, axis=-1)

    d = np.sqrt(np.sum((grid - center) ** 2, axis=-1))
    pos_mask = d < pos_radius
    neg_mask = d < neg_radius

    if label == 1:
        mask = pos_mask
    else:
        mask = neg_mask

    k = mask.shape[0]
    di, dj = np.where(mask)
    i = ci + di - k // 2
    j = cj + dj - k // 2

    keep = (0 <= i) & (i < y.shape[0]) & (0 <= j) & (j < y.shape[1])
    i = i[keep]
    j = j[keep]

    y[i, j] = label


def label_points_to_mask(points, shape):
    y = torch.zeros(*shape) + np.nan
    for i, j, label in points:
        fill_label_region(y, i, j, label)
    return y


def init_model(mrc):
    model = load_model("resnet16")
    model = model.features
    model.fill()
    model.eval()

    with torch.no_grad():
        filter_values = model(torch.from_numpy(mrc).float().unsqueeze(0)).squeeze(0)
    filter_values = filter_values.numpy()

    x = filter_values.transpose([1, 2, 0])
    x = x.reshape(-1, x.shape[-1])
    x = torch.from_numpy(x).float()

    return x, label_points_to_mask([], mrc.shape)


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
