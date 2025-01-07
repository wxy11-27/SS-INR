import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_coord
from .edsr import make_edsr_baseline
from .mlp import MLP

class LIIF(nn.Module):

    def __init__(self, local_ensemble=True, feat_unfold=True, cell_decode=True,n_feats=0):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = make_edsr_baseline(n_feats=n_feats,rgb_range=255)

        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(imnet_in_dim,n_feats,[256, 256, 256, 256])


    def gen_feat(self, inp):
        #self.feat = self.encoder(inp)
        self.feat = inp
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            #将feat中对应位置的像素值填充到grid指定的位置
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            # feat [b,64,48,48]
            #unfold(input, kernel_size, dilation=1, padding=0, stride=1)，3×3相邻隐码的合并
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            # feat [b,64*9,48,48]
        #局部集成，扩大每个隐码的表示
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # [b,2,48,48]
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6) #clamp (MIN, VAL, MAX) 函数的作用是把一个值限制在一个上限和下限之间，当这个值超过大最小值和最值的范围时，在最小值和最大值之间选择一个值使用
                #grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)将input中对应位置的像素值填充到grid指定的位置，得到最终的输出
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1), #coord_.flip(-1)对角镜像翻转
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


class LIIF1(nn.Module):

    def __init__(self, local_ensemble=True, feat_unfold=True, cell_decode=True,n_feats=0):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = make_edsr_baseline(n_feats=n_feats,rgb_range=255)

        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(imnet_in_dim,n_feats,[256, 256, 256, 256])


    def gen_feat(self, inp):
        #self.feat = self.encoder(inp)
        self.feat = inp
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            #将feat中对应位置的像素值填充到grid指定的位置
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            # feat [b,64,48,48]
            #unfold(input, kernel_size, dilation=1, padding=0, stride=1)，3×3相邻隐码的合并
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            # feat [b,64*9,48,48]
        #局部集成，扩大每个隐码的表示
        if self.local_ensemble:
            # vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        # rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # [b,2,48,48]
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        #for vx in vx_lst:
        for vy in vy_lst:
            coord_ = coord.clone()
            #coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_[:, :, 1].clamp_(-1 + 1e-6, 1 - 1e-6) #clamp (MIN, VAL, MAX) 函数的作用是把一个值限制在一个上限和下限之间，当这个值超过大最小值和最值的范围时，在最小值和最大值之间选择一个值使用
            #grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)将input中对应位置的像素值填充到grid指定的位置，得到最终的输出
            q_feat = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1), #coord_.flip(-1)对角镜像翻转
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord - q_coord
            #rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]
            inp = torch.cat([q_feat, rel_coord], dim=-1)

            if self.cell_decode:
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

            bs, q = coord.shape[:2]
            pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
            preds.append(pred)

            area = torch.abs(rel_coord[:, :, 0] ) #abs绝对值
            areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        # if self.local_ensemble:
        #     t = areas[0]; areas[0] = areas[3]; areas[3] = t
        #     t = areas[1]; areas[1] = areas[2]; areas[2] = t
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[1]; areas[1] = t
            #t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)