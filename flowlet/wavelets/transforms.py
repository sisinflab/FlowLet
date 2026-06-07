# ---------------------------------------------------
# Code adapted from: https://github.com/pfriedri/wdm-3d/tree/main/DWT_IDWT
# Thanks to the authors for their work.
# ----------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import math
import pywt


# ---------------------------------------------------
# Wavelet Transform Implementation (DWT/IDWT)
# ---------------------------------------------------

class DWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1).transpose(dim0=2, dim1=3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0=2, dim1=3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0=2, dim1=3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0=2, dim1=3)
        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0=2, dim1=3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0=2, dim1=3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0=2, dim1=3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0=2, dim1=3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0=2, dim1=3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0=2, dim1=3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0=2, dim1=3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0=2, dim1=3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH, grad_HLL, grad_HLH, grad_HHL, grad_HHH):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), grad_HLL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_LH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), grad_HLH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_HL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), grad_HHL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_HH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), grad_HHH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None

class IDWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input_LLL, input_LLH, input_LHL, input_LHH, input_HLL, input_HLH, input_HHL, input_HHH, matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2)
        input_LL = torch.add(torch.matmul(matrix_Low_2.t(), input_LLL.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), input_HLL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        input_LH = torch.add(torch.matmul(matrix_Low_2.t(), input_LLH.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), input_HLH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        input_HL = torch.add(torch.matmul(matrix_Low_2.t(), input_LHL.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), input_HHL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        input_HH = torch.add(torch.matmul(matrix_Low_2.t(), input_LHH.transpose(dim0=2, dim1=3)), torch.matmul(matrix_High_2.t(), input_HHH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        input_L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        input_H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), input_L), torch.matmul(matrix_High_0.t(), input_H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1).transpose(dim0=2, dim1=3)
        grad_LH = torch.matmul(grad_L, matrix_High_1).transpose(dim0=2, dim1=3)
        grad_HL = torch.matmul(grad_H, matrix_Low_1).transpose(dim0=2, dim1=3)
        grad_HH = torch.matmul(grad_H, matrix_High_1).transpose(dim0=2, dim1=3)
        grad_LLL = torch.matmul(matrix_Low_2, grad_LL).transpose(dim0=2, dim1=3)
        grad_LLH = torch.matmul(matrix_Low_2, grad_LH).transpose(dim0=2, dim1=3)
        grad_LHL = torch.matmul(matrix_Low_2, grad_HL).transpose(dim0=2, dim1=3)
        grad_LHH = torch.matmul(matrix_Low_2, grad_HH).transpose(dim0=2, dim1=3)
        grad_HLL = torch.matmul(matrix_High_2, grad_LL).transpose(dim0=2, dim1=3)
        grad_HLH = torch.matmul(matrix_High_2, grad_LH).transpose(dim0=2, dim1=3)
        grad_HHL = torch.matmul(matrix_High_2, grad_HL).transpose(dim0=2, dim1=3)
        grad_HHH = torch.matmul(matrix_High_2, grad_HH).transpose(dim0=2, dim1=3)
        return grad_LLL, grad_LLH, grad_LHL, grad_LHH, grad_HLL, grad_HLH, grad_HHL, grad_HHH, None, None, None, None, None, None

class DWT_3D(nn.Module):
    def __init__(self, wavename='haar'):
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self._cached_matrices = {}
        self._last_input_shape = None

    def get_matrix(self, input_shape, device='cpu'):
        if input_shape == self._last_input_shape and device in self._cached_matrices:
            return self._cached_matrices[device]

        input_depth, input_height, input_width = input_shape[-3:]
        L1 = max(input_depth, input_height, input_width)
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:math.floor(input_height / 2), 0:(input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:math.floor(input_width / 2), 0:(input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:math.floor(input_depth / 2), 0:(input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(input_height - math.floor(input_height / 2)), 0:(input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(input_width - math.floor(input_width / 2)), 0:(input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(input_depth - math.floor(input_depth / 2)), 0:(input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1[:, (self.band_length_half - 1):end])
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1[:, (self.band_length_half - 1):end])
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]

        matrices = {
            'low_0': torch.tensor(matrix_h_0, dtype=torch.float32, device=device),
            'low_1': torch.tensor(matrix_h_1, dtype=torch.float32, device=device),
            'low_2': torch.tensor(matrix_h_2, dtype=torch.float32, device=device),
            'high_0': torch.tensor(matrix_g_0, dtype=torch.float32, device=device),
            'high_1': torch.tensor(matrix_g_1, dtype=torch.float32, device=device),
            'high_2': torch.tensor(matrix_g_2, dtype=torch.float32, device=device)
        }

        self._cached_matrices[device] = matrices
        self._last_input_shape = input_shape
        return matrices

    def forward(self, input):
        assert len(input.size()) == 5 # N, C, D, H, W
        matrices = self.get_matrix(input.size(), input.device)
        return DWTFunction_3D.apply(input, matrices['low_0'], matrices['low_1'], matrices['low_2'],
                                    matrices['high_0'], matrices['high_1'], matrices['high_2'])

class IDWT_3D(nn.Module):
    def __init__(self, wavename='haar'):
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self._cached_matrices = {}
        self._last_target_shape = None

    def get_matrix(self, target_shape, device='cpu'):
        if target_shape == self._last_target_shape and device in self._cached_matrices:
            return self._cached_matrices[device]

        target_depth, target_height, target_width = target_shape[-3:]
        L1 = max(target_depth, target_height, target_width)
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:math.floor(target_height / 2), 0:(target_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:math.floor(target_width / 2), 0:(target_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:math.floor(target_depth / 2), 0:(target_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(target_height - math.floor(target_height / 2)), 0:(target_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(target_width - math.floor(target_width / 2)), 0:(target_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(target_depth - math.floor(target_depth / 2)), 0:(target_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1[:, (self.band_length_half - 1):end])
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1[:, (self.band_length_half - 1):end])
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]

        matrices = {
            'low_0': torch.tensor(matrix_h_0, dtype=torch.float32, device=device),
            'low_1': torch.tensor(matrix_h_1, dtype=torch.float32, device=device),
            'low_2': torch.tensor(matrix_h_2, dtype=torch.float32, device=device),
            'high_0': torch.tensor(matrix_g_0, dtype=torch.float32, device=device),
            'high_1': torch.tensor(matrix_g_1, dtype=torch.float32, device=device),
            'high_2': torch.tensor(matrix_g_2, dtype=torch.float32, device=device)
        }

        self._cached_matrices[device] = matrices
        self._last_target_shape = target_shape
        return matrices

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == 5 # N, C, D/2, H/2, W/2
        target_depth = LLL.size()[-3] + HHH.size()[-3]
        target_height = LLL.size()[-2] + HHH.size()[-2]
        target_width = LLL.size()[-1] + HHH.size()[-1]
        target_shape = (LLL.size(0), LLL.size(1), target_depth, target_height, target_width)

        matrices = self.get_matrix(target_shape, LLL.device)
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     matrices['low_0'], matrices['low_1'], matrices['low_2'],
                                     matrices['high_0'], matrices['high_1'], matrices['high_2'])


dwt_3d = DWT_3D(wavename='haar')
idwt_3d = IDWT_3D(wavename='haar')