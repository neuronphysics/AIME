import math
import torch.nn.functional as F
import torch
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
from torch.nn import init
import torchvision


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def PositionalEncoding2d(d_model, height, width):
    """
    Generate a 2D positional Encoding
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    # https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    height = int(height)
    width = int(width)
    pe = torch.zeros((d_model, height, width)).to(device)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * (-(math.log(10000.0) / d_model)))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = torch.reshape(pe, (1, d_model * 2, height, width))
    return pe


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CrossAxialAttention(nn.Module):
    # inspired by this code https://github.com/aL3x-O-o-Hung/TransformerForUltrasoundNeedleTracking/blob/c3fc204076a250a901628a5ca87c235fe969a253/network.py#L274
    def __init__(self, d1, h1, w1, d2, h2, w2):
        super(CrossAxialAttention, self).__init__()
        self.d1 = d1
        self.h1 = h1
        self.w1 = w1
        self.d2 = d2
        self.h2 = h2 // 2
        self.w2 = w2 // 2
        self.pe1_h = PositionalEncoding2d(d1, self.h2, 1)
        self.pe2_h = PositionalEncoding2d(d2, self.h2, 1)
        self.pe1_w = PositionalEncoding2d(d1, 1, self.w2)
        self.pe2_w = PositionalEncoding2d(d2, 1, self.w2)
        # X:[20, 1024, 18, 24], context: [20, 100, 4, 4]
        self.query_conv_h = nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=1, bias=False)  # x
        self.key_conv_h = nn.Conv2d(in_channels=d2, out_channels=d2, kernel_size=1, bias=False)  # context
        self.value_conv_h = nn.Conv2d(in_channels=d2, out_channels=d2, kernel_size=1, bias=False)  # context
        self.query_conv_w = nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=1, bias=False)  # x
        self.key_conv_w = nn.Conv2d(in_channels=d2, out_channels=d2, kernel_size=1, bias=False)  # context
        self.value_conv_w = nn.Conv2d(in_channels=d2, out_channels=d2, kernel_size=1, bias=False)  # context
        self.softmax_h = nn.Softmax(dim=-1)
        self.softmax_w = nn.Softmax(dim=-1)
        self.conv = []
        self.conv.append(nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=1))
        self.conv.append(nn.BatchNorm2d(d2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.conv.append(nn.Sigmoid())
        # self.conv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv = nn.Sequential(*self.conv)
        self.upsample = torch.nn.UpsamplingNearest2d(size=(self.h2, self.w2))
        self.upsample_x = torch.nn.UpsamplingBilinear2d(size=(h2, w2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    def forward(self, context, x):
        x = x.to(self.device)
        context = context.to(self.device)
        attentions = []
        x_ = self.pool(x)
        # change the height and weight of the context to the size of x_
        context_ = self.upsample(context)
        batch_size1, C1, height1, width1 = x_.size()
        batch_size2, C2, height2, width2 = context_.size()
        x_ = x_.permute(0, 3, 1, 2).contiguous().view(-1, C1, height1, 1)
        context_ = context_.permute(0, 3, 1, 2).contiguous().view(-1, C2, height2, 1)
        context_ = self.pe1_h + context_
        x_ = self.pe2_h + x_

        batch_size1_, C1_, height1_, width1_ = x_.size()
        batch_size2_, C2_, height2_, width2_ = context_.size()
        q = self.query_conv_h(context_).view(batch_size2_, -1, height2_ * width2_)
        k = self.key_conv_h(x_).view(batch_size1_, -1, height1_ * width1_)
        v = self.value_conv_h(x_).view(batch_size1_, -1, height1_ * width1_)

        energy = torch.einsum('bid,bjd->bij', q, k)
        attention = self.softmax_h(energy)
        attentions.append(attention)
        out = torch.einsum('bjd,bij->bid', v, attention)

        out = out.view(batch_size1_, C2_, height1_, width1_)

        out = out.squeeze().view(batch_size1, width1, C2, height1).permute(0, 2, 3, 1)
        x_ = x_.squeeze().view(batch_size1, width1, C1, height1).permute(0, 2, 3, 1)
        # torch.Size([20, 12, 1024, 9])
        context_ = out
        # context_ = context_.permute(0,2,1,3).contiguous().view(-1,C2,1,width2_)
        context_ = self.pe1_w + context_
        x_ = self.pe2_w + x_
        batch_size1_, C1_, height1_, width1_ = x_.size()
        batch_size2_, C2_, height2_, width2_ = context_.size()
        q = self.query_conv_w(context_).view(batch_size2_, -1, height2_ * width2_)
        k = self.key_conv_w(x_).view(batch_size1_, -1, height1_ * width1_)
        v = self.value_conv_w(x_).view(batch_size1_, -1, height1_ * width1_)

        energy = torch.einsum('bid,bjd->bij', q, k)
        attention = self.softmax_w(energy)
        attentions.append(attention)
        out = torch.einsum('bjd,bij->bid', v, attention)
        out = out.view(batch_size1_, C2_, height1_, width1_)

        out = out.squeeze().reshape(batch_size1, height1, C2, width1).permute(0, 2, 1, 3)
        x_ = x_.squeeze().reshape(batch_size1, height1, C1, width1).permute(0, 2, 1, 3)

        out = self.conv(out)
        out = self.upsample_x(out * x_)
        # if the input is:[20, 1024, 18, 24], context:[20, 100, 4, 4]
        # out:[20, 1024, 18, 24], attention: [240, 100, 1024], x_:[20, 1024, 9, 12]
        return out, attentions, x_


class Expand(torch.nn.Module):
    def __init__(self, in_channels, e1_out_channles, e3_out_channles):
        super(Expand, self).__init__()
        self.conv_1x1 = torch.nn.Conv2d(in_channels, e1_out_channles, (1, 1))
        self.conv_3x3 = torch.nn.Conv2d(in_channels, e3_out_channles, (3, 3), padding=1)

    def forward(self, x):
        o1 = self.conv_1x1(x)
        o3 = self.conv_3x3(x)
        return torch.cat((o1, o3), dim=1)


class Fire(torch.nn.Module):
    """
      source:https://github.com/xin-w8023/SqueezeNet-PyTorch/blob/afd81660cebb82ca1770f747c8a247a9267d0015/fire.py
      Fire module in SqueezeNet
      out_channles = e1x1 + e3x3
      Eg.: input: ?xin_channelsx?x?
           output: ?x(e1x1+e3x3)x?x?
    """

    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()

        # squeeze 
        self.squeeze = torch.nn.Conv2d(in_channels, s1x1, (1, 1))
        self.sq_act = torch.nn.LeakyReLU(0.1)

        # expand
        self.expand = Expand(s1x1, e1x1, e3x3)
        self.ex_act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.sq_act(self.squeeze(x))
        x = self.ex_act(self.expand(x))
        return x


class SqueezeNet(torch.nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            torch.nn.ReLU(),
            Fire(in_channels=96, s1x1=16, e1x1=64, e3x3=64),
            Fire(in_channels=128, s1x1=16, e1x1=64, e3x3=64),
            Fire(in_channels=128, s1x1=32, e1x1=128, e3x3=128),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            Fire(in_channels=256, s1x1=32, e1x1=128, e3x3=128),
            Fire(in_channels=256, s1x1=48, e1x1=192, e3x3=192),
            Fire(in_channels=384, s1x1=48, e1x1=192, e3x3=192),
            Fire(in_channels=384, s1x1=64, e1x1=256, e3x3=256),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            Fire(in_channels=512, s1x1=64, e1x1=256, e3x3=256),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1)


class SqueezeUnetEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(SqueezeUnetEncoder, self).__init__()

        self.seq = SqueezeNet()
        # print("Printing from Encoder")
        # print(self.seq)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6 in paper
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, 1)  # fc7 in paper
        self.conv8 = DoubleConv(in_channels=1024, out_channels=hidden_dim)

    def forward(self, *input):
        x = input[0]
        # original image size:[100, 3, 96, 96]
        # print(len(self.seq(x)))
        # conv1=self.seq[:2](x)
        # [100, 96, 224, 224]
        conv1 = F.adaptive_avg_pool2d(self.seq.net[:2](x), [224, 224]).squeeze()
        # [100, 128, 112, 112]
        conv2 = F.adaptive_avg_pool2d(self.seq.net[2:6](conv1), [112, 112]).squeeze()
        # [100, 256, 56, 56]
        # conv3 = self.seq[6:8](conv2)
        conv3 = F.adaptive_avg_pool2d(self.seq.net[6:8](conv2), [56, 56]).squeeze()
        # [100, 256, 56, 56]
        # conv4 = self.seq[8:11](conv3)
        conv4 = F.adaptive_avg_pool2d(self.seq.net[8:11](conv3), [28, 28]).squeeze()
        # [100, 384, 28, 28]
        # conv5= self.seq[11:](conv4)
        conv5 = F.adaptive_avg_pool2d(self.seq.net[11:](conv4), [28, 28]).squeeze()
        # [100, 512, 28, 28]
        conv6 = self.conv6(conv5)
        # [100, 1024, 28, 28]
        conv7 = self.conv7(conv6)
        # [100, 1024, 28, 28]
        conv8 = self.conv8(conv7)
        # [100, 64, 28, 28]

        return conv1, conv2, conv3, conv4, conv5, conv7, conv8


class UnetEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(UnetEncoder, self).__init__()
        configure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'm', 512, 512, 512, 'm']
        self.seq = make_layers(configure, 3)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6 in paper
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, 1)  # fc7 in paper
        self.conv8 = DoubleConv(in_channels=1024, out_channels=hidden_dim)

    def forward(self, *input):
        x = input[0]
        conv1 = self.seq[:4](x)
        conv2 = self.seq[4:9](conv1)
        conv3 = self.seq[9:16](conv2)
        conv4 = self.seq[16:23](conv3)
        conv5 = self.seq[23:](conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        return conv1, conv2, conv3, conv4, conv5, conv7, conv8


class DecoderCell(nn.Module):
    def __init__(self, size, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.picanet = PicanetG(size, in_channel)
        elif mode == 'L':
            self.picanet = PicanetL(in_channel)
        elif mode == 'C':
            self.picanet = None
        else:
            assert 0
        if not mode == 'C':
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)

    def forward(self, *input):
        assert len(input) <= 2
        if input[1] is None:
            en = input[0]
            dec = input[0]
        else:
            en = input[0]
            dec = input[1]

        if dec.size()[2] * 2 == en.size()[2]:
            dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        elif dec.size()[2] != en.size()[2]:
            assert 0

        en = self.bn_en(en)
        en = F.relu(en)
        fmap = torch.cat((en, dec), dim=1)  # F
        fmap = self.conv1(fmap)
        fmap = F.relu(fmap)
        if not self.mode == 'C':
            # print(self.mode)
            fmap_att = self.picanet(fmap)  # F_att
            x = torch.cat((fmap, fmap_att), 1)
            x = self.conv2(x)
            x = self.bn_feature(x)
            dec_out = F.relu(x)
            _y = self.conv3(dec_out)
            _y = torch.sigmoid(_y)
        else:
            dec_out = self.conv2(fmap)
            _y = torch.sigmoid(dec_out)

        return dec_out, _y


def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class PicanetG(nn.Module):
    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 100)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.renet(x)
        ksize = kernel.size()
        CRA = CrossAxialAttention(ksize[1], ksize[2], ksize[3], size[1], size[2], size[3])
        x, attn_comp, x_down = CRA(kernel, x)
        # kernel = F.softmax(kernel, 1)
        # kernel = kernel.reshape(size[0], 100, -1)  # ([9, 100, 16])
        # x = F.unfold(x, [1, 1], padding=[3, 3])  # (x, [10, 10], dilation=[3, 3])
        # x = x.reshape(size[0], size[1], 10 * 10)
        # x = torch.matmul(x, kernel)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x


class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 8 * 8, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        ksize = kernel.size()
        CRA = CrossAxialAttention(ksize[1], ksize[2], ksize[3], size[1], size[2], size[3])
        x, attn_comp, x_down = CRA(kernel, x)
        # kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # print("Before unfold", x.shape)
        # x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # print("After unfold", x.shape)
        # x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        # print(x.shape, kernel.shape)
        # x = torch.mul(x, kernel)
        # x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        # print(x.size())
        return x


class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)

    def forward(self, *input):
        x = input[0]
        temp = []
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel

        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = self.conv(x)
        return x


########################################
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, padding, nonlinearity=None):
        """
            1. in_channels is the number of input channels to the first conv layer, 
            2. out_channels is the number of output channels of the first conv layer 
                and the number of input channels to the second conv layer
        """
        super(ResidualBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                bias=False)

        )
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nl)
        layers.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                bias=False)

        )
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nl)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = out + x
        # each residual block doesn't wrap (res_x + x) with an activation function
        # as the next block implement ReLU as the first layer
        return out


class ResidualBlock_deconv(nn.Module):
    def __init__(self, channel, kernel_size, stride, padding, nonlinearity=None):
        super(ResidualBlock_deconv, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.conv1 = nn.ConvTranspose2d(channel, channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nl
        self.conv2 = nn.ConvTranspose2d(channel, channel, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + res
        return out


class LinearResidual(nn.Module):
    def __init__(self, input_feature, nonlinearity=None):
        super(LinearResidual, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        fn = []
        fn.append(
            nn.Linear(input_feature, input_feature)
        )
        fn.append(
            nn.BatchNorm1d(input_feature, affine=True)
        )
        fn.append(nl)
        fn.append(
            nn.Linear(input_feature, input_feature)
        )
        fn.append(
            nn.BatchNorm1d(input_feature, affine=True)
        )
        fn.append(nl)
        self.fn = nn.Sequential(*fn)

    def forward(self, x):
        return self.fn(x) + x


#########################
def build_grid(resolution, device):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


"""Adds soft positional embedding with learnable projection."""


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class SlotAttention(nn.Module):
    """
    https://arxiv.org/abs/2006.15055
    https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
    """

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


##########################
class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


class AdaBoundW(Optimizer):
    """Implements AdaBound algorithm with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBoundW, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBoundW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)

        return loss
