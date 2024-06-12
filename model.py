import torch
from torch import nn
import torch.nn.functional as F
from modules import ConvSC, Inception

def stride_generator(  N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self, C_in, C_out, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_out, stride=strides[0]),
            *[ConvSC(C_out, C_out, stride=s) for s in strides[1:]])

    def forward(self, x):
        enc1 = self.enc[0](x)       #output of 1st layer is used for skip connetion with decoder
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    def __init__(self, C_in, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)

        self.dec = nn.Sequential(
            *[ConvSC(C_in, C_in, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(C_in, C_in, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_in, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid+enc1)
        Y = self.readout(Y)
        return Y

class Observer(nn.Module):
    def __init__(self, in_c, N_h, h_c, N_T, T_c, incep_ker=[3, 5, 7, 11], groups=8):
        super(Observer, self).__init__()

        self.h_c=h_c
        self.T_c=T_c
        #h-inverse module
        self.N_h = N_h
        h_inv_layers = [Inception(C_in=in_c, C_hid=h_c // 2, C_out=h_c, incep_ker=incep_ker, groups=groups)]
        if  N_h > 1:
            for i in range(1, N_h - 1):
                h_inv_layers.append(Inception(h_c, h_c // 2, h_c, incep_ker=incep_ker, groups=groups))
            h_inv_layers.append(Inception(h_c, h_c // 2, h_c, incep_ker=incep_ker, groups=groups))
        self.h_inv = nn.Sequential(*h_inv_layers)

        #T module
        self.N_T = N_T
        T_layers = [Inception(C_in=h_c, C_hid=T_c // 2, C_out=T_c, incep_ker=incep_ker, groups=groups)]
        if N_T > 1:
            for i in range(1, N_T - 1):
                T_layers.append(Inception(T_c, T_c // 2, T_c, incep_ker=incep_ker, groups=groups))
            T_layers.append(Inception(T_c, T_c // 2, T_c, incep_ker=incep_ker, groups=groups))
        self.T = nn.Sequential(*T_layers)

        #T_inverse module
        T_inv_layers = [Inception(T_c, T_c // 2, T_c, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            T_inv_layers.append(
                Inception(2 * T_c, T_c // 2, T_c, incep_ker=incep_ker, groups=groups))
        T_inv_layers.append(Inception(2 * T_c, T_c // 2, h_c, incep_ker=incep_ker, groups=groups))
        self.T_inv = nn.Sequential(*T_inv_layers)

        # h module
        h_layers = []
        for i in range(0, N_h - 1):
            h_layers.append(Inception(2 * h_c, h_c // 2, h_c, incep_ker=incep_ker, groups=groups))
        h_layers.append(Inception(2 * h_c, h_c // 2, in_c, incep_ker=incep_ker, groups=groups))
        self.h = nn.Sequential(*h_layers)

        #Parameterize A and initialize it with kaiming_uniform_
        self.weight_A = nn.Parameter(torch.empty(T_c,16,16))     #give the shape manually for different dataset.
        nn.init.kaiming_uniform_(self.weight_A)

        #Parameterize B, using CNN or Linear layer
        self.B_conv=nn.Conv2d(in_c, T_c, kernel_size=3, padding=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)  # [16, 80, 8, 8]      时间模块中，应该把batch展开去计算

        #State estimation with h-inverse module
        h_skips = []
        z=x
        for i in range(self.N_h):
            z = self.h_inv[i](z)
            h_skips.append(z)
            #if i < self.N_h - 1:
            #    h_skips.append(z)

        z_in= z.reshape(B, self.h_c, H, W)

        # Dynamic transformation with T module
        t_skips = []
        xi = z
        for i in range(self.N_T):
            xi = self.T[i](xi)
            t_skips.append(xi)

        xi_in =xi.reshape(B, self.T_c, H, W)
        #xi_in = xi

        #Future forecasting with linear state prediction
        #let self.weight_A input a sigmoid function to ensure every element is between 0 and 1   -softmax  - sigmoid
        self.weight_A_norm = F.sigmoid(self.weight_A)
        xi_pre = self.weight_A_norm*xi + self.B_conv(x)
        xi_pred = xi_pre.reshape(B, self.T_c, H, W)

        # Dynamic inverting with T_inverse module
        z_pre = self.T_inv[0](xi_pre)
        for i in range(1, self.N_T):
            j = i + 1
            z_pre = self.T_inv[i](torch.cat([z_pre, t_skips[-j]], dim=1))

        z_pred = z_pre.reshape(B, self.h_c, H, W)

        # Latent output with h module
        x_pre = z_pre
        for i in range(0, self.N_h):
            j=i+1
            x_pre = self.h[i](torch.cat([x_pre, h_skips[-j]], dim=1))

        x_pred = x_pre.reshape(B, T, C, H, W)  # [16, 10, 8, 8, 8]

        return x_pred, z_in, xi_in, xi_pred, z_pred

class ST_Observer(nn.Module):
    def __init__(self, in_shape, N_S, en_de_c, N_h, h_c, N_T, T_c, groups=1, incep_ker=list((3, 5, 7, 11))):
        super(ST_Observer, self).__init__()
        T, C, H, W = in_shape  # [4, 2, 32, 32]

        self.SpatialEncoder = Encoder(C_in=C, C_out=en_de_c, N_S =N_S)
        self.Observer = Observer(in_c=en_de_c*T, N_h=N_h, h_c=h_c, N_T=N_T, T_c=T_c, incep_ker=incep_ker, groups=groups)
        self.SpatialDecoder = Decoder(C_in=en_de_c, C_out=C, N_S=N_S)

    def forward(self, y_his):
        #reshape y from (B, T, C, H, W) to (B*T, C, H, W)
        B, T, C, H, W = y_his.shape  # B=16, T=10, C=1, H=64, W=64
        y = y_his.view(B * T, C, H, W)

        #spatial encoder
        embeding, skip = self.SpatialEncoder(y)
        _, C_, H_, W_ = embeding.shape  # C_=64, H_=16, W_=16

        #ST Observer for forecasting
        x = embeding.view(B, T, C_, H_, W_)
        x_in = x

        pred_embeding, z_in, xi_in, xi_pred,  z_pred = self.Observer(x)

        x_pred=pred_embeding
        #reshape pred_embeding
        pred_embeding = pred_embeding.reshape(B * T, C_, H_, W_)

        #decode the prediction
        Prediction = self.SpatialDecoder(pred_embeding, skip)
        Prediction = Prediction.reshape(B, T, C, H, W)

        return Prediction, x_in,  z_in, xi_in, xi_pred,  z_pred, x_pred