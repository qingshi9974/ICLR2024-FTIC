from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel
from compressai.layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import os
from compressai.ops import LowerBound
from range_coder import RangeEncoder, RangeDecoder
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange
from .entropy_models import GsnConditionalLocScaleShift
from .tca import TCA_EntropyModel
from timm.models.layers import  DropPath
import math


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x




def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, idx, split_size=8, dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
   
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size*2, self.split_size*2
        elif idx == 1:
            H_sp, W_sp = self.split_size//2, self.split_size//2
        elif idx == 2:
            H_sp, W_sp = self.split_size//2, self.split_size*2
        elif idx == 3:
            H_sp, W_sp = self.split_size*2, self.split_size//2
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        window_size = [H_sp,W_sp]
        self.attn_drop = nn.Dropout(attn_drop)
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)


    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)
      
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
   
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x



class Swin_FDWA(nn.Module):
   
    def __init__(self, dim,  num_heads,
                 window_size=8,  window_size_fm=16, shift_size=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 4
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                WindowAttention(
                    dim//self.branch_num ,  idx = i,
                    split_size=window_size, num_heads=num_heads//self.branch_num , dim_out=dim//self.branch_num ,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])

      
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fm = WindowFrequencyModulation(dim,window_size_fm)
    
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm2 = norm_layer(dim)


    def calculate_mask(self, H, W,split_size=[8,8]):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1)).cpu() # 1 H W 1 idx=0
        shift_size =(split_size[0]//2,split_size[1]//2)

        h_slices_0 = (slice(0, -split_size[0]),
                    slice(-split_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices_0 = (slice(0, -split_size[1]),
                    slice(-split_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))

        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt +=1
 

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // split_size[0], split_size[0], W // split_size[1], split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, split_size[0], split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, split_size[0] * split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
    
        return attn_mask_0

    def forward(self, x, x_size):
        H , W = x_size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C

        if self.shift_size>0:
            qkv = qkv.view(3, B, H, W, C)
     
            qkv0,qkv1,qkv2,qkv3= qkv.chunk(4,4)
            qkv_0 = torch.roll(qkv0, shifts=(-self.split_size,-self.split_size), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, L, C//4)

            qkv_1 = torch.roll(qkv1, shifts=(-self.split_size//4,-self.split_size//4), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, L, C//4)
         
            qkv_2 = torch.roll(qkv2, shifts=(-self.split_size//4,-self.split_size), dims=(2, 3))
            qkv_2 = qkv_2.view(3, B, L, C//4)
         
            qkv_3 = torch.roll(qkv3, shifts=(-self.split_size,-self.split_size//4), dims=(2, 3))
            qkv_3 = qkv_3.view(3, B, L, C//4)
          
            x1_shift = self.attns[0](qkv_0, H, W)
            x2_shift = self.attns[1](qkv_1, H, W)
            x3_shift = self.attns[2](qkv_2, H, W)
            x4_shift = self.attns[3](qkv_3, H, W)
                
            x1 = torch.roll(x1_shift, shifts=(self.split_size, self.split_size), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.split_size//4, self.split_size//4), dims=(1, 2))
            x3 = torch.roll(x3_shift, shifts=(self.split_size//4, self.split_size), dims=(1, 2))
            x4 = torch.roll(x4_shift, shifts=(self.split_size, self.split_size//4), dims=(1, 2))

            x1 = x1.view(B, L, C//4)
            x2 = x2.view(B, L, C//4)
            x3 = x3.view(B, L, C//4)
            x4 = x4.view(B, L, C//4)
            # Concat
            attened_x = torch.cat([x1,x2,x3,x4], dim=2)
           
         
        else:
            qkv0,qkv1,qkv2,qkv3= qkv.chunk(4,3)
            x1 = self.attns[0](qkv0, H, W).view(B, L, C//4)
            x2 = self.attns[1](qkv1, H, W).view(B, L, C//4)
            x3 = self.attns[2](qkv2, H, W).view(B, L, C//4)
            x4 = self.attns[3](qkv3, H, W).view(B, L, C//4)
            # Concat
            attened_x = torch.cat([x1,x2,x3,x4], dim=2)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.fm(self.ffn(self.norm2(x)),H,W)

        return x



class WindowFrequencyModulation(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.ratio = 1
        self.complex_weight= nn.Parameter(torch.cat((torch.ones(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32),\
        torch.zeros(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32)),dim=-1))
    def forward(self, x, H,W,spatial_size=None):
        B,L,C = x.shape
   
        x = x.view(B,H,W,self.ratio*C)
        B, H,W, C = x.shape

        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)

        x = x.to(torch.float32)
        
        x= torch.fft.rfft2(x,dim=(3, 4), norm='ortho')
      
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 p1) (w2 p2) c ')

        x = x.view(B, -1, C)
        return x

class FAT_Block(nn.Module):
    def __init__(self, trans_dim, head_dim, window_size, window_size_fm,drop_path, type='W',hyper=False):
        """ SwinTransformer and Conv Block
        """
        super(FAT_Block, self).__init__()
        self.trans_dim = trans_dim
        self.head_dim = head_dim

        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']

        self.trans_block =  Swin_FDWA(
                dim=trans_dim,
                num_heads=head_dim,
                window_size=window_size,
                window_size_fm=window_size_fm,
                shift_size=0 if (type=='W') else window_size//2)
    
        
        self.conv1_1 = nn.Conv2d(self.trans_dim, self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.trans_dim, self.trans_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        trans_x = self.conv1_1(x)
        b,c,h,w = trans_x.shape
        trans_x = Rearrange('b c h w -> b (h w)c')(trans_x)
        trans_x = self.trans_block(trans_x,(h,w))
        trans_x = Rearrange('b (h w) c -> b c h w',h=h,w=w)(trans_x)
        
        res = self.conv1_2(trans_x)
        
        x = x + res
        return x


class FrequencyAwareTransFormer(CompressionModel):
    def __init__(self, config=[2, 2, 2,2, 2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128,  M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.tca_depth = 12
        self.tca_ratio = 4
        self.lower_bound = 0.01
        
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        N1 = 96 
        N2 = 144
        N3 = 256

        self.m_down1 = [FAT_Block( N1, self.head_dim[0], self.window_size,self.window_size*2, dpr[i], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(N1, N2, stride=2)]
                      
        self.m_down2 = [FAT_Block( N2, self.head_dim[1], self.window_size, self.window_size*2,dpr[i+2], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(N2, N3, stride=2)]
                      
        self.m_down3 = [FAT_Block(N3, self.head_dim[2], self.window_size, self.window_size*2,dpr[i+4], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                       [conv3x3(N3, M, stride=2)]

       
        dpr =dpr[::-1]
        self.m_up1 = [FAT_Block( N3, self.head_dim[3], self.window_size, self.window_size*2,dpr[i], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(N3, N2, 2)]
        self.m_up2 = [FAT_Block( N2, self.head_dim[4], self.window_size,self.window_size*2, dpr[i+2], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(N2, N1, 2)]
        self.m_up3 = [FAT_Block(N1, self.head_dim[5], self.window_size,self.window_size*2, dpr[i+4], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(N1, 3, 2)]
   
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, N1, 2)] + self.m_down1 + self.m_down2 + self.m_down3) 
        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [FAT_Block(N*2, 32, 2,4, 0, 'W' if not i%2 else 'SW',True) 
                      for i in range(config[0])] + \
                      [conv3x3(2*N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, N3, 2)] + \
            self.ha_down1
        )

        self.hs_up1 = [FAT_Block( N*2, 32, 2,4, 0, 'W' if not i%2 else 'SW',True) 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up1
        )

        self.hs_up2 = [FAT_Block( N*2, 32, 2,4, 0, 'W' if not i%2 else 'SW',True) 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]


        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up2
        )
      
        self.tca= TCA_EntropyModel(dim=M,ratio=self.tca_ratio,depth=self.tca_depth,slices=self.num_slices)
        self.entropy_bottleneck  = EntropyBottleneck(192)
        self.gaussian_conditional=    GsnConditionalLocScaleShift(
            num_scales=256, num_means=100, min_scale=self.lower_bound, tail_mass=(2 ** (-8))
        )
    
    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        scales_hyper = self.h_scale_s(z_hat)
        means_hyper  = self.h_mean_s(z_hat)        
 
        y_hat = ste_round(y)
        means,scales,lrp = self.tca(torch.cat((means_hyper,scales_hyper),dim=1), y_hat)
        _, y_likelihoods =  self.gaussian_conditional(y, scales, means)

        lrp = 0.5 * torch.tanh(lrp)
        y_hat += lrp
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net
    def compress(self, x,output_path):
        
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        hyper = torch.cat((latent_means,latent_scales),dim=1)
        y_hat_coded = torch.round(y)
        channel_per_slices = self.M//self.num_slices
        minmax = np.maximum(abs(torch.round(y.cpu()).max()), abs(torch.round(y.cpu()).min()))
        minmax = int(np.maximum(minmax, 1))
        encoder = RangeEncoder(output_path)
        samples = np.arange(0, minmax*2+1)
        lower_bound =  LowerBound(1e-9)
        lower_bound.bound = lower_bound.bound.to(x.device)

        means,scales,_ = self.tca(hyper,y_hat_coded)
        for slice_index in range(self.num_slices):
            mu = means[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            scale = scales[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            y_slice = y[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            y_hat_slice = torch.round(y_slice)

            for h_idx in range(y_shape[0]):
                for w_idx in range(y_shape[1]):
                    for c_idx in range(channel_per_slices):

                        pmf = self._likelihood(torch.tensor(samples).to(x.device),scale[0][c_idx][h_idx][w_idx],means=mu[0][c_idx][h_idx][w_idx]+minmax)
                        pmf = lower_bound(pmf)
                        pmf_clip = np.clip(np.array(pmf.cpu()), 1.0/65536, 1.0)   
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                        
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]
                    
                        symbol = int(y_hat_slice[0, c_idx,h_idx, w_idx] + minmax )
                        encoder.encode([symbol], cdf)
     
        encoder.close()
        y_size = os.path.getsize(output + '.bin')
        return {"z_strings":  z_strings, "z_shape": z.size()[-2:],"y_size":y_size,"minmax":minmax}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        lower_bound =  LowerBound(self.lower_bound)
        lower_bound.bound = lower_bound.bound.to(inputs.device)
        scales = lower_bound(scales)
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
      
        return half * torch.erfc(const * inputs)

    def decompress(self, z_strings, minmax, z_shape,output_path):
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        hyper = torch.cat((latent_means,latent_scales),dim=1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_hat_coded = torch.zeros((1,self.M,z_hat.shape[2]*4,z_hat.shape[3]*4)).to(z_hat.device)
        lrp_coded = torch.zeros((1,self.M,z_hat.shape[2]*4,z_hat.shape[3]*4)).to(z_hat.device)
        channel_per_slices = self.M//self.num_slices
        decoder = RangeDecoder(output_path)
        samples = np.arange(0, minmax*2+1)

        lower_bound =  LowerBound(1e-9)
        lower_bound.bound = lower_bound.bound.to(z_hat.device)
        for slice_index in range(self.num_slices):
            means,scales,lrps = self.tca(hyper,y_hat_coded)
            mu = means[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            scale = scales[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            lrp = lrps[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            y_hat_slice = torch.zeros_like(mu).to(z_hat.deivce)
            for h_idx in range(y_shape[0]):
                for w_idx in range(y_shape[1]):
                    for c_idx in range(channel_per_slices):

                        pmf = self._likelihood(torch.tensor(samples),scale[0][c_idx][h_idx][w_idx],means=mu[0][c_idx][h_idx][w_idx]+minmax)
                        pmf = lower_bound(pmf)
                        pmf_clip = np.clip(np.array(pmf.cpu()), 1.0/65536, 1.0)   
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                       
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]
                       
                        y_hat_slice[0,  c_idx,h_idx, w_idx] = decoder.decode(1, cdf)[0] - minmax 

            y_hat_coded[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices] = y_hat_slice
            lrp_coded[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices] = lrp


        lrp_coded = 0.5 * torch.tanh(lrp_coded)
  
        y_hat= y_hat_coded+lrp_coded
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}