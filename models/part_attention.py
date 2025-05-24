import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSplitAttention(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        self.group_channels = channels // groups
        # 为每组分别建 Query/Key/Value 投影
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        # 输出融合
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        # 可学习缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # [B, 3C, H, W] -> 分为 Q,K,V 各 [B, C, H, W]
        qkv = self.qkv(x).reshape(B, 3, self.groups, self.group_channels, H*W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # 每项 [B, G, Cg, N]
        # 重排列： [B, G, N, Cg] 用于注意力
        q = q.permute(0,1,3,2)
        k = k.permute(0,1,3,2)
        # 计算注意力
        attn = torch.softmax(q @ k.transpose(-2,-1) / (self.group_channels**0.5), dim=-1)  # [B,G,N,N]
        # 应用到 V
        v = v.permute(0,1,3,2)  # [B,G,N,Cg]
        out = (attn @ v).permute(0,1,3,2).reshape(B, C, H, W)  # [B,C,H,W]
        out = self.proj(out)
        # 加上残差
        return x + self.gamma * out


def INF(B, H, W, device):
    """
    行方向注意力时的 -∞ 掩码。
    构造形状 [B*W, H, H]，对角线填 -inf，以避免 Query 与自身 Key 的注意力。
    """
    # 创建一个 H×H 的对角矩阵，主对角线为 -inf
    return -torch.diag(torch.full((H,), float('inf'), device=device)).unsqueeze(0).repeat(B*W, 1, 1)

class CrissCrossAttention(nn.Module):
    """
    Criss-Cross Attention Module
    输入 x: [B, C, H, W]
    输出: [B, C, H, W] （与输入同形）
    """
    def __init__(self, in_dim):
        super().__init__()
        # Query/Key/Value 的 1×1 投影
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, bias=False)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,     kernel_size=1, bias=False)
        self.softmax    = nn.Softmax(dim=3)
        # 残差融合系数，初始为 0
        self.gamma      = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device

        # 1. 生成 Q, K, V
        proj_query = self.query_conv(x)   # [B, Cq, H, W]
        proj_key   = self.key_conv(x)     # [B, Cq, H, W]
        proj_value = self.value_conv(x)   # [B, C,  H, W]

        # 2. 行方向（H）上的重排与注意力
        # 将 (B, Cq, H, W) -> (B*W, Cq, H), 然后调换到 (B*W, H, Cq)
        proj_query_H = proj_query.permute(0, 3, 1, 2)\
                                   .contiguous()\
                                   .view(B*W, -1, H)\
                                   .permute(0, 2, 1)
        proj_key_H   = proj_key.permute(0, 3, 1, 2)\
                                 .contiguous()\
                                 .view(B*W, -1, H)
        proj_value_H = proj_value.permute(0, 3, 1, 2)\
                                   .contiguous()\
                                   .view(B*W, -1, H)

        # 3. 列方向（W）上的重排与注意力
        proj_query_W = proj_query.permute(0, 2, 1, 3)\
                                   .contiguous()\
                                   .view(B*H, -1, W)\
                                   .permute(0, 2, 1)
        proj_key_W   = proj_key.permute(0, 2, 1, 3)\
                                 .contiguous()\
                                 .view(B*H, -1, W)
        proj_value_W = proj_value.permute(0, 2, 1, 3)\
                                   .contiguous()\
                                   .view(B*H, -1, W)

        # 4. 计算行和列的能量矩阵（attention scores）
        # 行方向加 -∞ 掩码，防止自注意
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + INF(B, H, W, device))\
                     .view(B, W, H, H)\
                     .permute(0, 2, 1, 3)  # -> [B, H, W, H]
        energy_W = torch.bmm(proj_query_W, proj_key_W)\
                     .view(B, H, W, W)    # -> [B, H, W, W]

        # 5. 拼接行、列能量并归一化
        # 在最后一维上 concat 成 [B, H, W, H+W]
        energy = torch.cat([energy_H, energy_W], dim=3)
        attn   = self.softmax(energy)       # [B, H, W, H+W]

        # 6. 分离行、列注意力，并加权 V
        attn_H = attn[:, :, :, :H]          \
                   .permute(0, 2, 1, 3)    \
                   .contiguous()          \
                   .view(B*W, H, H)       # [B*W, H, H]
        attn_W = attn[:, :, :, H:]         \
                   .contiguous()          \
                   .view(B*H, W, W)       # [B*H, W, W]

        out_H = torch.bmm(proj_value_H, attn_H.permute(0,2,1))\
                     .view(B, W, C, H)\
                     .permute(0, 2, 3, 1)          # -> [B, C, H, W]
        out_W = torch.bmm(proj_value_W, attn_W.permute(0,2,1))\
                     .view(B, H, C, W)\
                     .permute(0, 2, 1, 3)          # -> [B, C, H, W]

        # 7. 残差融合
        out = self.gamma * (out_H + out_W) + x
        return out


class DualAttentionBlock(nn.Module):
    def __init__(self, channels, groups=4, recursion=2):
        super().__init__()
        self.chan_attn = ChannelSplitAttention(channels, groups)
        self.spat_attn = CrissCrossAttention(channels)
        self.recursion = recursion
        self.conv = nn.Conv2d(2*channels, channels, 1)
    def forward(self, x):
        # 通道分块注意力
        x_chan = self.chan_attn(x)
        # 空间分块注意力，递归多次
        x_spat = x
        for _ in range(self.recursion):
            x_spat = self.spat_attn(x_spat)
        # 将两路特征按通道拼接后，再投影回原始通道数
        out = torch.cat([x_chan, x_spat], dim=1)  # [B,2C,H,W]
        #out = nn.Conv2d(2*x.shape[1], x.shape[1], 1).to(x.dtype)(out)
        out = self.conv(out)
        return out

x = torch.randn([2,1024,14,14])
test = DualAttentionBlock(1024)
x = test(x)
print(x.shape)