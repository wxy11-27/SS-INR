import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from liif import utils as liif
from liif.liif import LIIF
from liif.liif import LIIF1
from data_loader import build_datasets1,build_datasets2

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#################################################################
class SSINR(nn.Module):
    def __init__(self, 
                 arch,
                 scale_ratio,
                 n_select_bands, 
                 n_bands,
                 size
                 ):
        """Load the pretrained ResNet and replace top fc layer."""
        super(SSINR, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))
        self.size = size
        self.conv_fus = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spat = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spec = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_bands+5, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_bands+5, n_bands, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(n_bands, n_select_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(n_select_bands, n_select_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


        self.downsample = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
        )
        # liif model
        self.liif_model = LIIF(n_feats=self.n_bands)
        self.liif_model1 = LIIF1(n_feats=128)
        self.ca = ChannelAttention(n_bands)
        self.sa = SpatialAttention()

    def lrhr_interpolate(self, x_lr, x_hr):

        gap_bands = self.n_bands / (self.n_select_bands - 1.0)
        for i in range(0, self.n_select_bands - 1):
            x_lr[:, int(gap_bands * i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands - 1), ::] = x_hr[:, self.n_select_bands - 1, ::]

        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

        return edge



    def forward(self, x_lr, x_hr,x_ref):

        n_bands =self.n_bands


        ###################空间上采样######################
        upsample_times = 4
        hr_coord1 = liif.make_coord([x_lr.shape[-2] * upsample_times, x_lr.shape[-1] * upsample_times]).unsqueeze(0).to(
            x_lr.device)  # [86, 86] -> [86*86, 2]
        cell = torch.ones_like(hr_coord1)
        cell[:, 0] *= 2 / x_lr.shape[-2]
        cell[:, 1] *= 2 / x_lr.shape[-1]
        h0 = self.liif_model(x_lr, hr_coord1, cell).permute(0, 2, 1).contiguous(). \
            view(1, self.n_bands, x_lr.shape[-2] * upsample_times, x_lr.shape[-1] * upsample_times)

        h = self.conv4(torch.cat((h0,x_hr),1))


        # ################################空间R################################
        l1 = build_datasets1(img=h,size=128, n_select_bands=5, scale_ratio=4).to(x_lr.device)
        r1 = l1 - x_lr
        upsample_times = 4
        hr_coord1 = liif.make_coord([r1.shape[-2] * upsample_times, x_lr.shape[-1] * upsample_times]).unsqueeze(0).to(
            r1.device)
        cell = torch.ones_like(hr_coord1)
        cell[:, 0] *= 2 / r1.shape[-2]
        cell[:, 1] *= 2 / r1.shape[-1]
        x_lr_liif = self.liif_model(r1, hr_coord1, cell).permute(0, 2, 1).contiguous(). \
            view(1, self.n_bands, r1.shape[-2] * upsample_times, r1.shape[-1] * upsample_times)
        R1 = x_lr_liif
        # ##############################光谱 R #######################################
        h3 = build_datasets2(img=h, size=128, n_select_bands=5, scale_ratio=4).to(x_lr.device)
        r2 = h3 - x_hr
        ####光谱20
        r2 = r2.permute(0, 2, 3, 1).contiguous()
        hr_coord2 = liif.make_coord([128, 20]).unsqueeze(0).to(
            x_hr.device)  # [86, 86] -> [86*86, 2]
        cell1 = torch.ones_like(hr_coord2)
        cell1[:, 0] *= 2 / cell1.shape[-2]
        cell1[:, 1] *= 2 / cell1.shape[-1]
        x_lr_liif1 = self.liif_model1(r2, hr_coord2, cell1).permute(0, 2, 1).contiguous(). \
            view(1, 128, 128, 20).permute(0, 3, 1, 2).contiguous()
        r2 = x_lr_liif1
        #####光谱50
        r2 = r2.permute(0, 2, 3, 1).contiguous()
        hr_coord2 = liif.make_coord([128, 50]).unsqueeze(0).to(
            x_hr.device)  # [86, 86] -> [86*86, 2]
        cell1 = torch.ones_like(hr_coord2)
        cell1[:, 0] *= 2 / cell1.shape[-2]
        cell1[:, 1] *= 2 / cell1.shape[-1]
        x_lr_liif1 = self.liif_model1(r2, hr_coord2, cell1).permute(0, 2, 1).contiguous(). \
            view(1, 128, 128, 50).permute(0, 3, 1, 2).contiguous()
        r2 = x_lr_liif1
        #####光谱n_bands
        r2 = r2.permute(0, 2, 3, 1).contiguous()
        hr_coord2 = liif.make_coord([128, n_bands]).unsqueeze(0).to(
            x_hr.device)  # [86, 86] -> [86*86, 2]
        cell1 = torch.ones_like(hr_coord2)
        cell1[:, 0] *= 2 / cell1.shape[-2]
        cell1[:, 1] *= 2 / cell1.shape[-1]
        x_lr_liif1 = self.liif_model1(r2, hr_coord2, cell1).permute(0, 2, 1).contiguous(). \
            view(1, 128, 128, n_bands).permute(0, 3, 1, 2).contiguous()
        R2 = x_lr_liif1

        x = self.conv_fus(h+R1+R2)
        x = x + self.conv_spat(x)
        x = x + self.conv_spec(x)


        return x, 0, 0, 0, 0 ,0


