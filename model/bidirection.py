# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import numbers
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def Win_Dila(x, win_size):
    """
    :param x: B, H, W, C
    :param
    :return: y: B, H, W, C
    """
    n_win = win_size * 2
    B, H, W, C = x.shape
    x = x.reshape(-1, (H // n_win), n_win, (W // n_win), n_win, C)  # B x n1 x n_win x n2 x n_win x C
    x = x.permute(0, 1, 3, 2, 4, 5)  # B x n1 x n2 x n_win x n_win x C

    xt = torch.zeros_like(x)
    x0 = x[:, :, :, 0::2, 0::2, :]
    x1 = x[:, :, :, 0::2, 1::2, :]
    x2 = x[:, :, :, 1::2, 0::2, :]
    x3 = x[:, :, :, 1::2, 1::2, :]

    xt[:, :, :, 0:n_win//2, 0:n_win//2, :] = x0  # B n/2 n/2 d2c
    xt[:, :, :, 0:n_win//2, n_win//2:n_win, :] = x1  # B n/2 n/2 d2c
    xt[:, :, :, n_win//2:n_win, 0:n_win//2, :] = x2  # B n/2 n/2 d2c
    xt[:, :, :, n_win//2:n_win, n_win//2:n_win, :] = x3  # B n/2 n/2 d2c
    xt = xt.permute(0, 1, 3, 2, 4, 5)
    xt = xt.reshape(-1, H, W, C)

    return xt

def Win_ReDila(x, win_size):
    """
    :param x: B, H, W, C
    :param
    :return: y: B, H, W, C
    """
    n_win = win_size * 2
    B, H, W, C = x.shape
    x = x.reshape(-1, (H // n_win), n_win, (W // n_win), n_win, C)  # B x n1 x n_win x n2 x n_win x C
    x = x.permute(0, 1, 3, 2, 4, 5)  # B x n1 x n2 x n_win x n_win x C

    xt = torch.zeros_like(x)
    xt[:, :, :, 0::2, 0::2, :] = x[:, :, :, 0:n_win//2, 0:n_win//2, :]
    xt[:, :, :, 0::2, 1::2, :] = x[:, :, :, 0:n_win//2, n_win//2:n_win, :]
    xt[:, :, :, 1::2, 0::2, :] = x[:, :, :, n_win//2:n_win, 0:n_win//2, :]
    xt[:, :, :, 1::2, 1::2, :] = x[:, :, :, n_win//2:n_win, n_win//2:n_win, :]

    xt = xt.permute(0, 1, 3, 2, 4, 5)
    xt = xt.reshape(-1, H, W, C)

    return xt


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample_unc(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_unc, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Stage(nn.Module):
    def __init__(self, dim=32, input_resolution=(16, 16), num_heads=8, window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.window_size <= input_resolution[0], "input_resolution should be larger than window_size"
        assert 0 <= self.window_size <= input_resolution[1], "input_resolution should be larger than window_size"

        self.attn1 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn2 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn3 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        attn_windows = self.attn1(x_windows)  # nW*B, window_size*window_size, C
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp1(self.norm2(x)))

        shortcut = x
        x = self.norm3(x)
        x = x.view(B, H, W, C)
        shifted_x = Win_Dila(x, self.window_size)
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn2(x_windows)  # nW*B, window_size*window_size, C
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = Win_ReDila(shifted_x, self.window_size)
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp2(self.norm4(x)))

        shortcut = x
        x = self.norm5(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        attn_windows = self.attn3(x_windows)  # nW*B, window_size*window_size, C
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp3(self.norm6(x)))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class XCAttention(nn.Module):
    def __init__(self, dim, num_heads, win, bias):
        super(XCAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.win = win
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, C, H // self.win, self.win, W // self.win, self.win)
        q_win = q.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.win * self.win)

        k = k.view(B, C, H // self.win, self.win, W // self.win, self.win)
        k_win = k.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.win * self.win)

        v = v.view(B, C, H // self.win, self.win, W // self.win, self.win)
        v_win = v.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.win * self.win)

        q = rearrange(q_win, 'n (head c) win -> n head c win', head=self.num_heads)
        k = rearrange(k_win, 'n (head c) win -> n head c win', head=self.num_heads)
        v = rearrange(v_win, 'n (head c) win -> n head c win', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'n head c (h w) -> n (head c) h w', w=self.win)
        out = out.view(B, H // self.win, W // self.win, C, self.win, self.win)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        out = self.project_out(out)

        return out


##########################################################################
class XCABlock(nn.Module):
    def __init__(self, dim, num_heads, win, ffn_expansion_factor, bias, LayerNorm_type):
        super(XCABlock, self).__init__()

        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = XCAttention(dim, num_heads, win, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Direction1(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=31,
                 embed_dim=96, num_heads=[],
                 window_size=8, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        '''
        :param patch_size: for the embed conv
        :param in_chans: for the embed conv

        '''
        super(Direction1, self).__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

        # split image into non-overlapping patches
        downsample = PatchMerging
        self.downsample1 = downsample(dim=self.embed_dim, input_resolution=(self.img_size, self.img_size),
                                      norm_layer=norm_layer)
        self.downsample2 = downsample(dim=self.embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),
                                      norm_layer=norm_layer)

        self.stage1 = Stage(dim=self.embed_dim, input_resolution=(self.img_size, self.img_size), num_heads=num_heads[0], window_size=window_size)
        self.stage2 = Stage(dim=self.embed_dim * 2, input_resolution=(self.img_size//2, self.img_size//2), num_heads=num_heads[1], window_size=window_size)
        self.stage3 = Stage(dim=self.embed_dim * 4, input_resolution=(self.img_size//4, self.img_size//4), num_heads=num_heads[2], window_size=window_size)


    def forward(self, x):
        B, _, H, _ = x.shape
        x = self.conv(x)  # B x em x H x W
        x = rearrange(x, 'B C H W -> B (H W) C ')
        stage1_out = self.stage1(x)
        stage1_out_shape = rearrange(stage1_out, 'B (H W) C -> B C H W', H=H)  # B x C x H x W
        stage1_out = self.downsample1(stage1_out)  # B (H/2 W/2) C
        H = H // 2

        stage2_out = self.stage2(stage1_out)
        stage2_out_shape = rearrange(stage2_out, 'B (H W) C -> B C H W', H=H)  # B x 2C x H/2 x W/2
        stage2_out = self.downsample2(stage2_out)  # B (H/4 W/4) C

        H = H // 2
        stage3_out = self.stage3(stage2_out)
        stage3_out_shape = rearrange(stage3_out, 'B (H W) C -> B C H W', H=H)  # B x 4C x H/4 x W/4

        return stage1_out_shape, stage2_out_shape, stage3_out_shape

class Direction2(nn.Module):
    def __init__(self, inp_channels=31, dim=32, num_heads=[8, 8, 8], group=8, ffn_expansion_factor=2.66,  LayerNorm_type = 'WithBias', bias=False):
        super(Direction2, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.stage1 = XCABlock(dim=dim, num_heads=num_heads[0], win=group, ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type)
        self.up1_2 = Upsample_unc(int(dim))

        self.stage2 = XCABlock(dim=dim, num_heads=num_heads[0], win=group, ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.up2_3 = Upsample_unc(int(dim))

        self.stage3 = XCABlock(dim=dim, num_heads=num_heads[0], win=group, ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type)


    def forward(self, x):
        emb = self.patch_embed(x)
        stage_output1 = self.stage1(emb)  # B x dim x h x w
        stage_output1_up = self.up1_2(stage_output1)  # B x dim/2 x 2h x 2w

        stage_output2 = self.stage2(stage_output1_up)   # B x dim/2 x 2h x 2w
        stage_output2_up = self.up2_3(stage_output2)  # B x dim/4 x 4h x 4w

        stage_output3 = self.stage3(stage_output2_up)  # B x dim/4 x 4h x 4w

        return stage_output3, stage_output2, stage_output1

class Merge(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans1=34, in_chans2=31, embed_dim=48, num_heads1=[8, 8, 8],window_size=8, group=8, mlp_ratio=4.,dim=32,
                 num_heads2=[8, 8, 8], ffn_expansion_factor=2.66,  LayerNorm_type = 'WithBias', bias=False
                 ):
        super(Merge, self).__init__()
        self.direction1 = Direction1(img_size=img_size, patch_size=patch_size, in_chans=in_chans1, embed_dim=embed_dim, num_heads=num_heads1, window_size=window_size, mlp_ratio=mlp_ratio)
        self.direction2 = Direction2(inp_channels=in_chans2, dim=dim, num_heads=num_heads2, group=group, ffn_expansion_factor=ffn_expansion_factor,  LayerNorm_type=LayerNorm_type, bias=bias)
        self.merge1 = nn.Sequential(nn.Conv2d(embed_dim * 4 + dim, dim, kernel_size=3, padding=1, bias=bias),
                                    nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=bias),
                                    nn.LeakyReLU(0.2, bias))

        self.up1 = Upsample_unc(int(dim))

        self.merge2 = nn.Sequential(nn.Conv2d(embed_dim * 2 + dim + dim, dim, kernel_size=3, padding=1, bias=bias),
                                    nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=bias),
                                    nn.LeakyReLU(0.2, bias))

        self.up2 = Upsample_unc(int(dim))

        self.merge3 = nn.Sequential(nn.Conv2d(embed_dim + dim + dim, in_chans2, kernel_size=3, padding=1, bias=bias),
                                    nn.Conv2d(in_chans2, in_chans2, kernel_size=5, padding=2, bias=bias),)

    def forward(self, x, y):
        D1_stage1, D1_stage2, D1_stage3 = self.direction1(x)
        D2_stage3, D2_stage2, D2_stage1 = self.direction2(y)

        merge1 = self.merge1(torch.cat((D1_stage3, D2_stage1), 1))
        merge2 = self.merge2(torch.cat((D1_stage2, D2_stage2, self.up1(merge1)), 1))
        output = self.merge3(torch.cat((D1_stage1, D2_stage3, self.up2(merge2)), 1))

        return output


if __name__ == '__main__':
    input1 = torch.randn(1, 34, 64, 64).cuda()
    input2 = torch.randn(1, 31, 16, 16).cuda()

    # up = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=None)
    # print(up(input2).shape)
    # exit()
    # t = Bidirection(img_size=64, patch_size=1, in_chans=34, embed_dim=32, num_heads=[8, 8, 8], window_size=4).cuda()
    # t = Direction2().cuda()
    t = Merge(img_size=64, patch_size=1, in_chans1=34, in_chans2=31, embed_dim=48, num_heads1=[8, 8, 8],
                       window_size=8, group=8,  mlp_ratio=4.,dim=48, num_heads2=[8, 8, 8],  ffn_expansion_factor=2.66,  LayerNorm_type = 'WithBias',
                       bias=False).cuda()
    output = t(input1, input2)
    num_params = 0
    for param in Merge(img_size=64, patch_size=1, in_chans1=34, in_chans2=31, embed_dim=48, num_heads1=[8, 8, 8],
                       window_size=8, group=8,  mlp_ratio=4.,dim=48, num_heads2=[8, 8, 8],  ffn_expansion_factor=2.66,  LayerNorm_type = 'WithBias',
                       bias=False).parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('Bottleneck', num_params / 1e6))
    print(output.shape)
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    print(flop_count_table(FlopCountAnalysis(t, (input1, input2))))






