
import cv2
import torch
import torch.nn as nn
import numpy as np
def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels

class CannyFilter(nn.Module):
    #https://github.com/DCurro/CannyEdgePytorch/blob/master/net_canny.py
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=True):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        for param in self.gaussian_filter.parameters():
            param.requires_grad = False

        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        for param in self.sobel_filter_x.parameters():
            param.requires_grad = False
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        for param in self.sobel_filter_y.parameters():
            param.requires_grad = False
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        for param in self.directional_filter.parameters():
            param.requires_grad = False
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        for param in self.hysteresis.parameters():
            param.requires_grad = False
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        """
        Differentiable variant:
          - No rounding of orientation
          - Soft non-max suppression
          - Soft thresholds + hysteresis
        """
        B, C, H, W = img.shape
        device, dtype = img.device, img.dtype

        # tensors for steps
        blurred = torch.zeros((B, C, H, W), device=device, dtype=dtype)
        grad_x = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
        grad_y = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        # gaussian + sobel per channel
        for c in range(C):
            blurred_c = self.gaussian_filter(img[:, c:c+1])   # [B,1,H,W]
            blurred[:, c:c+1] = blurred_c
            grad_x = grad_x + self.sobel_filter_x(blurred_c)
            grad_y = grad_y + self.sobel_filter_y(blurred_c)

        # average over channels
        grad_x, grad_y = grad_x / C, grad_y / C

        # gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # [B,1,H,W]

        # continuous orientation (no rounding)
        grad_orientation = torch.atan2(grad_y, grad_x + 1e-6)          # radians
        grad_orientation = grad_orientation * (360.0 / np.pi) + 180.0  # degrees in [0, 360)
        grad_orientation = torch.remainder(grad_orientation, 360.0)

        # directional responses for NMS
        directional = self.directional_filter(grad_magnitude)          # [B,8,H,W]

        # -------- Soft orientation assignment (instead of rounding) --------
        dirs = torch.arange(0, 360, 45, device=device, dtype=dtype).view(1, 8, 1, 1)  # [1,8,1,1]
        theta = grad_orientation  # [B,1,H,W] degrees
        theta_exp = theta.repeat(1, 8, 1, 1)  # [B,8,H,W]

        # circular distance in degrees, in [-180, 180]
        delta = ((theta_exp - dirs + 180.0) % 360.0) - 180.0

        sigma_orient = 30.0  # controls how sharp the orientation assignment is
        orient_logits = - (delta ** 2) / (2.0 * sigma_orient ** 2)
        orientation_weights = torch.softmax(orient_logits, dim=1)  # [B,8,H,W]

        # -------- Soft non-maximum suppression --------
        # Pair opposite directions: (0,4), (1,5), (2,6), (3,7)
        pos = directional[:, 0:4, :, :]      # [B,4,H,W]
        neg = directional[:, 4:8, :, :]      # [B,4,H,W]
        pair_min = torch.minimum(pos, neg)   # [B,4,H,W], like min(pos,neg) in original

        pair_weights = orientation_weights[:, 0:4, :, :] + orientation_weights[:, 4:8, :, :]
        # aggregate over direction pairs into a scalar "NMS score"
        directional_score = (pair_min * pair_weights).sum(dim=1, keepdim=True)  # [B,1,H,W]

        tau_nms = 1.0  # temperature for NMS sharpness
        gate_nms = torch.sigmoid(directional_score / tau_nms)

        # thinned edges (still continuous)
        thin_edges = grad_magnitude * gate_nms  # [B,1,H,W]

        # -------- Soft thresholds (no hard > / ==) --------
        if low_threshold is not None:
            tau_thr = 1.0

            # low_gate ≈ 1 if thin_edges >> low_threshold, else ≈ 0
            low_gate = torch.sigmoid((thin_edges - low_threshold) / tau_thr)

            if high_threshold is not None:
                # high_gate ≈ 1 if thin_edges >> high_threshold
                high_gate = torch.sigmoid((thin_edges - high_threshold) / tau_thr)

                # emulate 0 / 0.5 / 1 but smoothly:
                #   strong: low≈1, high≈1 → 1.0
                #   weak:   low≈1, high≈0 → 0.5
                #   none:   low≈0, high≈0 → 0.0
                thin_edges = 0.5 * low_gate + 0.5 * high_gate

                if hysteresis:
                    # soft hysteresis:
                    # weak_prob ~ "weak edges" (low but not high)
                    weak_prob = low_gate * (1.0 - high_gate)
                    # propagate strong edges via conv and a soft gate
                    conv_edges = self.hysteresis(thin_edges)
                    tau_hyst = 1.0
                    gate_conv = torch.sigmoid((conv_edges - 1.0) / tau_hyst)
                    thin_edges = high_gate + weak_prob * gate_conv
            else:
                # only low threshold: smoothly suppress low magnitudes
                thin_edges = thin_edges * low_gate

        return thin_edges
