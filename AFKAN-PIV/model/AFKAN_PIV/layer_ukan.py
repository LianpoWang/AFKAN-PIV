from model.AFKAN_PIV.UKAN import *
from model.AFKAN_PIV.ScConv import *

class AllFeatureFusion(nn.Module):
    def __init__(self, args, dim=128):
        super().__init__()
        self.trans1 = nn.Conv2d(2,dim,1,1,bias=False)
        self.trans2 = nn.Conv2d(100,dim,1,1,bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            SRU(dim * 2),
            CRU(dim * 2),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()
        self.SRU = SRU(dim)
        self.CRU = CRU(dim)

    def forward(self, flow, corr):
        flow = self.trans1(flow)
        corr = self.trans2(corr)
        fus0 = torch.cat([flow, corr], dim=1)
        att = self.conv_atten(self.avg_pool(fus0))
        output = fus0 * att
        output = self.conv_redu(output)
        att = self.conv1(self.SRU(self.CRU((flow)))) + self.conv2(self.SRU(self.CRU((corr))))
        att = self.nonlin(att)
        output = output * att
        return output

class Second_part(nn.Module):
    def __init__(self, args, hdim=128, cdim=128):
        #net: hdim, inp: cdim
        super(Second_part, self).__init__()
        self.args = args
        self.encoder = AllFeatureFusion(args, dim=cdim)
        self.refine = []
        self.refine.append(UKan_Hybrid(1,2*cdim+hdim, [1,1,1,1],[],2,0.0))
        self.refine = nn.ModuleList(self.refine)#将卷积列表转化为模型，方便反向传播

    def forward(self, net, context, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)#移动特征解码器
        context = torch.cat([context, motion_features], dim=1)
        t = torch.tensor([0], dtype=torch.long).to(context.device)
        for blk in self.refine:
            net = blk(torch.cat([net, context], dim=1), t)
        return net

GRAD_CLIP = 0.1

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
