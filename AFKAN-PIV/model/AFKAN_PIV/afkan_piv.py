import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from model.AFKAN_PIV.layer_ukan import Second_part
from core.corr import CorrBlock
from core.utils.utils import coords_grid, InputPadder
from core.extractor import ResNetFPN
from core.utils.layer import conv1x1, conv3x3
import math
from huggingface_hub import PyTorchModelHubMixin

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class GaussianFourierFeatureTransform_B(torch.nn.Module):
    def __init__(self, num_input_channels, B, mapping_size=256, scale=10):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = B * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())
        batches, channels, width, height = x.shape
        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)
        x = x @ self._B.to(x.device)
        x = x.view(batches, width, height, self._mapping_size)
        x = x.permute(0, 3, 1, 2)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class AFKAN_PIV(
    nn.Module,
    PyTorchModelHubMixin,
):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.dim * 2
        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d,init_weight=True)
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d,init_weight=True)
            self.update_block = Second_part(args, hdim=args.dim, cdim=args.dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords2 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []

        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H // 8, W // 8, device=image1.device)
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.args.iters > 0:
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        if test_mode == False:
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.args.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.args.var_max
                    var_min = self.args.var_min

                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2,dim=2)
                nf_predictions.append(nf_loss)

            return flow_predictions
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None}
