import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(in_channels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1_1 = nn.Conv2d(32, 64, 5, stride=1, padding=int((5 - 1) / 2))
        self.down1_2 = nn.Conv2d(64, 64, 5, stride=1, padding=int((5 - 1) / 2))

        self.down2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=int((3 - 1) / 2))
        self.down2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=int((3 - 1) / 2))

        self.down3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=int((3 - 1) / 2))
        self.down3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=int((3 - 1) / 2))

        self.down4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=int((3 - 1) / 2))
        self.down4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=int((3 - 1) / 2))

        self.down5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=int((3 - 1) / 2))
        self.down5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=int((3 - 1) / 2))

        self.up1_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.up1_2 = nn.Conv2d(2 * 512, 512, 3, stride=1, padding=1)

        self.up2_1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.up2_2 = nn.Conv2d(2 * 256, 256, 3, stride=1, padding=1)

        self.up3_1 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.up3_2 = nn.Conv2d(2 * 128, 128, 3, stride=1, padding=1)

        self.up4_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.up4_2 = nn.Conv2d(2 * 64, 64, 3, stride=1, padding=1)

        self.up5_1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.up5_2 = nn.Conv2d(2 * 32, 32, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(32, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)

        s2 = F.avg_pool2d(s1, 2)
        s2 = F.leaky_relu(self.down1_1(s2), negative_slope=0.1)
        s2 = F.leaky_relu(self.down1_2(s2), negative_slope=0.1)

        s3 = F.avg_pool2d(s2, 2)
        s3 = F.leaky_relu(self.down2_1(s3), negative_slope=0.1)
        s3 = F.leaky_relu(self.down2_2(s3), negative_slope=0.1)

        s4 = F.avg_pool2d(s3, 2)
        s4 = F.leaky_relu(self.down3_1(s4), negative_slope=0.1)
        s4 = F.leaky_relu(self.down3_2(s4), negative_slope=0.1)

        s5 = F.avg_pool2d(s4, 2)
        s5 = F.leaky_relu(self.down4_1(s5), negative_slope=0.1)
        s5 = F.leaky_relu(self.down4_2(s5), negative_slope=0.1)

        x = F.avg_pool2d(s5, 2)
        x = F.leaky_relu(self.down5_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.down5_2(x), negative_slope=0.1)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.leaky_relu(self.up1_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.up1_2(torch.cat((x, s5), 1)), negative_slope=0.1)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.leaky_relu(self.up2_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.up2_2(torch.cat((x, s4), 1)), negative_slope=0.1)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.leaky_relu(self.up3_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.up3_2(torch.cat((x, s3), 1)), negative_slope=0.1)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.leaky_relu(self.up4_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.up4_2(torch.cat((x, s2), 1)), negative_slope=0.1)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.leaky_relu(self.up5_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.up5_2(torch.cat((x, s1), 1)), negative_slope=0.1)

        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x


class SuperSlomo():
    """
    interpolate frames between two inference
    """

    def __init__(self, arbitrary_time_flow_interpolation_unet, backward_warping, I_0, I_1, device, F_0_1, F_1_0,
                 train=False):
        self.arbitrary_time_flow_interpolation_unet = arbitrary_time_flow_interpolation_unet
        self.I_0 = I_0.to(device)
        self.I_1 = I_1.to(device)
        self.device = device
        self.F_0_1 = F_0_1.to(device)
        self.F_1_0 = F_1_0.to(device)
        self.backward_warping = backward_warping
        self.train = train

    def forward(self, speedup, t=-1):
        """
        :param speedup: if you want to convert 25-fps to 75-fps, you should specify speedup=3
        :return: list of Tensor contains all the interpolated frames in order
        """
        interpolated = []

        """
        simple modification to be adapted in training
        """
        if self.train:
            speedup = 2

        for i in range(1, speedup):
            """
            interpolate at time T = t
            """
            if not self.train:
                t = i / speedup
            else:
                t = t.item()

            F_t_0_hat = -(1 - t) * t * self.F_0_1 + t ** 2 * self.F_1_0
            F_t_1_hat = (1 - t) ** 2 * self.F_0_1 - t * (1 - t) * self.F_1_0
            g_t_0_hat = self.backward_warping(self.I_0, F_t_0_hat)
            g_t_1_hat = self.backward_warping(self.I_1, F_t_1_hat)
            interpolation_outputs = self.arbitrary_time_flow_interpolation_unet(torch.cat((self.I_0, self.I_1,
                                                                                           self.F_0_1, self.F_1_0,
                                                                                           F_t_1_hat, F_t_0_hat,
                                                                                           g_t_0_hat,
                                                                                           g_t_1_hat), dim=1))
            """
            training the residual
            """
            F_t_0 = interpolation_outputs[:, :2, :, :] + F_t_0_hat
            F_t_1 = interpolation_outputs[:, 2:4, :, :] + F_t_1_hat
            V_t_0 = torch.sigmoid(interpolation_outputs[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            g_0 = self.backward_warping(self.I_0, F_t_0)
            g_1 = self.backward_warping(self.I_1, F_t_1)
            Z = ((1 - t) * V_t_0 + t * V_t_1)
            I_t_hat = ((1 - t) * V_t_0 * g_0 + t * V_t_1 * g_1) / Z
            interpolated.append(I_t_hat)
        if self.train:
            return g_t_0_hat, g_t_1_hat, I_t_hat, self.backward_warping(self.I_0, self.F_1_0), self.backward_warping(
                self.I_1, self.F_0_1)
        else:
            return interpolated


class BackwardWarping(nn.Module):

    def __init__(self, width, height, device):
        super(BackwardWarping, self).__init__()
        grid_X, grid_Y = np.meshgrid(np.arange(width), np.arange(height))
        self.W = width
        self.H = height
        self.grid_X = torch.tensor(grid_X, requires_grad=False, device=device)
        self.grid_Y = torch.tensor(grid_Y, requires_grad=False, device=device)

    def forward(self, img, flow):
        h = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.grid_X.unsqueeze(0).expand_as(h).float() + h
        y = self.grid_Y.unsqueeze(0).expand_as(v).float() + v
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        grid = torch.stack((x, y), dim=3)
        """
        Bi-Linear Interpolation
        """
        interpolated = torch.nn.functional.grid_sample(img, grid)
        return interpolated
