"""Joint supervision used by the source training script."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=0.001):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class ColorCorrectnessLoss(nn.Module):

    def __init__(self):
        super(ColorCorrectnessLoss, self).__init__()

    def forward(self, x, y):
        eps = 1e-08
        loss_cos = 1.0 - F.cosine_similarity(x, y, dim=1, eps=eps)
        b, c, h, w = x.shape
        mean_rgb_est = torch.mean(x.view(b, c, -1), dim=2)
        mean_rgb_gt = torch.mean(y.view(b, c, -1), dim=2)
        loss_mean = F.mse_loss(mean_rgb_est, mean_rgb_gt)
        return torch.mean(loss_cos) + loss_mean

class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.register_buffer('kernel', torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1))
        self.loss = nn.L1Loss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice = nn.Sequential()
        for i in range(30):
            self.slice.add_module(str(i), vgg[i])
        self.slice.eval()
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = (x - 0.5) / 0.5
        y_vgg = (y - 0.5) / 0.5
        loss = F.l1_loss(self.slice(x_vgg), self.slice(y_vgg))
        return loss


class JointLoss(nn.Module):
    """1.0 pixel + 0.05 edge + 0.5 color + 0.01 VGG-19 (first 30 layers)."""
    def __init__(self):
        super().__init__()
        self.pixel = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.color = ColorCorrectnessLoss()
        self.perceptual = PerceptualLoss()

    def forward(self, output, target):
        return (self.pixel(output, target) + 0.05 * self.edge(output, target)
                + 0.5 * self.color(output, target)
                + 0.01 * self.perceptual(output, target))
