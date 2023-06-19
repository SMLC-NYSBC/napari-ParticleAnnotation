import torch.nn.functional as F
from scipy.optimize import minimize
import numpy as np
from topaz.model.factory import load_model
import torch
import torch.nn as nn

from topaz.model.utils import insize_from_outsize


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


def initialize_model(mrc):
    model = load_model("resnet16")
    model = model.features
    model.fill()
    model.eval()

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
        filter_values = filter_values.numpy()  # C, W, H

        x = filter_values.transpose([1, 2, 0])  # W, H, C

    x = x.reshape(-1, x.shape[-1])  # L, C
    x = torch.from_numpy(x).float()
    y = torch.zeros(len(x)) + np.nan
    return x, y


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


class LinearClassifier(nn.Module):
    """A simple convolutional layer without non-linear activation."""

    def __init__(self, features):
        """
        Args:
            features (:obj:): the sizes associated with the layer

        Attributes:
            features (:obj:)
        """
        super(LinearClassifier, self).__init__()
        self.features = features
        self.classifier = nn.Conv3d(features.latent_dim, 1, 1)

    @property
    def width(self):
        return self.features.width

    @property
    def latent_dim(self):
        return self.features.latent_dim

    def fill(self, stride=1):
        return self.features.fill(stride=stride)

    def unfill(self):
        self.features.unfill()

    def forward(self, x):
        """Applies the classifier to an input.

        Args:
            x (np.ndarray): the image from which features are extracted and classified

        Returns:
            z (np.ndarray): output of the classifer
        """
        z = self.features(x)
        y = self.classifier(z)
        return y


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()

        if "pooling" in kwargs:
            pooling = kwargs["pooling"]
            if pooling == "max":
                kwargs["pooling"] = MaxPool

        modules = self.make_modules(**kwargs)
        self.features = nn.Sequential(*modules)

        self.width = insize_from_outsize(modules, 1)
        self.pad = False

    ## make property for num_features !!

    def fill(self, stride=1):
        for mod in self.features.children():
            if hasattr(mod, "fill"):
                stride *= mod.fill(stride)
        self.pad = True
        return stride

    def unfill(self):
        for mod in self.features.children():
            if hasattr(mod, "unfill"):
                mod.unfill()
        self.pad = False

    def set_padding(self, pad):
        self.pad = pad
        # for mod in self.features:
        #    if hasattr(mod, 'set_padding'):
        #        mod.set_padding(pad)

    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)  # add channels dim
        if self.pad:  ## add (width-1)//2 zeros to edges of x
            p = self.width // 2
            x = F.pad(x, (p, p, p, p))
        z = self.features(x)
        return z
        # return self.classifier(z)[:,0] # remove channels dim


class ResNet16(ResNet):
    def make_modules(
        self,
        units=[32, 64, 128],
        bn=True,
        dropout=0.0,
        activation=nn.ReLU,
        pooling=None,
        **kwargs,
    ):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2 * units, 4 * units]

        self.num_features = units[-1]
        self.stride = 1
        if pooling is None:
            self.stride = 2
        stride = self.stride

        modules = [
            BasicConv3d(1, units[0], 7, bn=bn, activation=activation),
            ResidA(
                units[0],
                units[0],
                units[0],
                stride=stride,
                bn=bn,
                activation=activation,
            ),
        ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))  # , inplace=True))

        modules += [
            ResidA(units[0], units[0], units[0], bn=bn, activation=activation),
            ResidA(units[0], units[0], units[0], bn=bn, activation=activation),
            ResidA(units[0], units[0], units[0], bn=bn, activation=activation),
            ResidA(
                units[0],
                units[0],
                units[1],
                stride=stride,
                bn=bn,
                activation=activation,
            ),
        ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))  # , inplace=True))

        modules += [
            ResidA(units[1], units[1], units[1], bn=bn, activation=activation),
            ResidA(units[1], units[1], units[1], bn=bn, activation=activation),
            BasicConv3d(units[1], units[2], 5, bn=bn, activation=activation),
        ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))  # , inplace=True))

        self.latent_dim = units[-1]

        return modules


class BasicConv3d(nn.Module):
    def __init__(
        self, nin, nout, kernel_size, dilation=1, stride=1, bn=False, activation=nn.ReLU
    ):
        super(BasicConv3d, self).__init__()

        bias = not bn
        self.conv = nn.Conv3d(
            nin, nout, kernel_size, dilation=dilation, stride=stride, bias=bias
        )
        if bn:
            self.bn = nn.BatchNorm3d(nout)
        self.act = activation(inplace=True)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.og_dilation = dilation
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            p = self.dilation * (self.kernel_size // 2)
            self.conv.padding = (p, p)
            self.padding = p
        else:
            self.conv.padding = (0, 0)
            self.padding = 0

    def fill(self, stride):
        self.conv.dilation = (self.og_dilation * stride, self.og_dilation * stride)
        self.conv.stride = (1, 1)
        self.conv.padding = (
            self.conv.padding[0] * stride,
            self.conv.padding[1] * stride,
        )
        self.dilation *= stride
        return self.stride

    def unfill(self):
        stride = self.dilation // self.og_dilation
        self.conv.dilation = (self.og_dilation, self.og_dilation)
        self.conv.stride = (self.stride, self.stride)
        self.conv.padding = (
            self.conv.padding[0] // stride,
            self.conv.padding[1] // stride,
        )
        self.dilation = self.og_dilation

    def forward(self, x):
        y = self.conv(x)
        if hasattr(self, "bn"):
            y = self.bn(y)
        return self.act(y)


class ResidA(nn.Module):
    def __init__(
        self, nin, nhidden, nout, dilation=1, stride=1, activation=nn.ReLU, bn=False
    ):
        super(ResidA, self).__init__()

        self.bn = bn
        bias = not bn

        if nin != nout:
            self.proj = nn.Conv3d(nin, nout, 1, stride=stride, bias=False)

        self.conv0 = nn.Conv3d(nin, nhidden, 3, bias=bias)
        if self.bn:
            self.bn0 = nn.BatchNorm3d(nhidden)
        self.act0 = activation(inplace=True)

        self.conv1 = nn.Conv3d(
            nhidden, nout, 3, dilation=dilation, stride=stride, bias=bias
        )
        if self.bn:
            self.bn1 = nn.BatchNorm3d(nout)
        self.act1 = activation(inplace=True)

        self.kernel_size = 2 * dilation + 3
        self.stride = stride
        self.dilation = 1
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            self.conv0.padding = (1, 1, 1)
            self.conv1.padding = self.conv1.dilation
            self.padding = self.kernel_size // 2
        else:
            self.conv0.padding = (0, 0, 0)
            self.conv1.padding = (0, 0, 0)
            self.padding = 0

    def fill(self, stride):
        self.conv0.dilation = (stride, stride)
        self.conv0.padding = (
            self.conv0.padding[0] * stride,
            self.conv0.padding[1] * stride,
        )
        self.conv1.dilation = (
            self.conv1.dilation[0] * stride,
            self.conv1.dilation[1] * stride,
        )
        self.conv1.stride = (1, 1, 1)
        self.conv1.padding = (
            self.conv1.padding[0] * stride,
            self.conv1.padding[1] * stride,
        )
        if hasattr(self, "proj"):
            self.proj.stride = (1, 1, 1)
        self.dilation = self.dilation * stride
        return self.stride

    def unfill(self):
        self.conv0.dilation = (1, 1, 1)
        self.conv0.padding = (
            self.conv0.padding[0] // self.dilation,
            self.conv0.padding[1] // self.dilation,
        )
        self.conv1.dilation = (
            self.conv1.dilation[0] // self.dilation,
            self.conv1.dilation[1] // self.dilation,
        )
        self.conv1.stride = (self.stride, self.stride)
        self.conv1.padding = (
            self.conv1.padding[0] // self.dilation,
            self.conv1.padding[1] // self.dilation,
        )
        if hasattr(self, "proj"):
            self.proj.stride = (self.stride, self.stride)
        self.dilation = 1

    def forward(self, x):
        h = self.conv0(x)
        if self.bn:
            h = self.bn0(h)
        h = self.act0(h)

        y = self.conv1(h)

        # d2 = x.size(2) - y.size(2)
        # d3 = x.size(3) - y.size(3)
        # if d2 > 0 or d3 > 0:
        #    lb2 = d2//2
        #    ub2 = d2 - lb2
        #    lb3 = d3//2
        #    ub3 = d3 - lb3
        #    x = x[:,:,lb2:-ub2,lb3:-ub3]

        edge = self.conv0.dilation[0] + self.conv1.dilation[0]
        x = x[:, :, edge:-edge, edge:-edge]

        if hasattr(self, "proj"):
            x = self.proj(x)
        elif self.conv1.stride[0] > 1:
            x = x[:, :, :: self.stride, :: self.stride]

        y = y + x
        if self.bn:
            y = self.bn1(y)
        y = self.act1(y)

        return y


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.og_stride = stride
        self.dilation = 1
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            p = self.dilation * (self.kernel_size // 2)  # this is bugged in pytorch...
            # p = self.kernel_size//2
            self.pool.padding = (p, p, p)
            self.padding = p
        else:
            self.pool.padding = (0, 0, 0)
            self.padding = 0

    def fill(self, stride):
        self.pool.dilation = stride
        self.pool.padding = self.pool.padding * stride
        self.pool.stride = 1
        self.dilation = stride
        self.stride = 1
        return self.og_stride

    def unfill(self):
        self.pool.dilation = 1
        self.pool.padding = self.pool.padding // self.dilation
        self.pool.stride = self.og_stride
        self.dilation = 1
        self.stride = self.og_stride

    def forward(self, x):
        return self.pool(x)
