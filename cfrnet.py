"""CFRNet and its CFRB, RISE and CCFR components."""
import torch
from torch import nn
from torch.nn import functional as F
from backbone import BlockSequence, EBlock, DBlock


class RISE(nn.Module):
    """Real–Imaginary Squeeze Encoding of signed spectral coefficients."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, spectrum_components):
        return self.pool(spectrum_components)


class CCFR(nn.Module):
    """Jointly predict real/imaginary gains shared across frequency positions."""
    def __init__(self, channels):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(2 * channels, channels // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 2, 2 * channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, descriptor):
        return self.predictor(descriptor)


class CFRB(nn.Module):
    """Channel-wise Frequency Response Bottleneck with residual projection."""
    def __init__(self, channels):
        super().__init__()
        self.rise = RISE()
        self.ccfr = CCFR(channels)
        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        spectrum = torch.fft.rfft2(x, norm="ortho")
        components = torch.cat((spectrum.real, spectrum.imag), dim=1)
        gains = self.ccfr(self.rise(components))
        real, imag = (components * gains).chunk(2, dim=1)
        restored = torch.fft.irfft2(
            torch.complex(real, imag), s=(height, width), norm="ortho"
        )
        return x + self.fusion(restored)



class CFRNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2, enc_blk_nums=(1, 2, 3), dec_blk_nums=(3, 1, 1), dilations=(1, 4, 9), extra_depth_wise=True):
        super(CFRNet, self).__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(BlockSequence(*[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks_enc = BlockSequence(*[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)])
        self.cfrb = CFRB(chan)
        self.middle_blks_dec = BlockSequence(*[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(BlockSequence(*[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)
        self.side_out = nn.Conv2d(in_channels=width * 2 ** len(self.encoders), out_channels=img_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, input, side_loss=False):
        _, _, H, W = input.shape
        input = self.check_image_size(input)
        x = self.intro(input)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x_light = self.middle_blks_enc(x)
        x_light = self.cfrb(x_light)
        if side_loss:
            out_side = self.side_out(x_light)
        x = self.middle_blks_dec(x_light)
        x = x + x_light
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)
        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W]
        if side_loss:
            return (out_side, out)
        else:
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)
        return x
