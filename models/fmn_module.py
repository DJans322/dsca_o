# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang
# 完全独立的前景调制网络模块，解耦所有依赖

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module, Identity, Sequential, Linear, Dropout, LayerNorm, BatchNorm1d, BatchNorm2d
from typing import Optional, Dict, Union, Sequence
from dataclasses import dataclass, field
from collections import OrderedDict
# --------------------------- 辅助组件（内联原代码库的子模块） --------------------------- #

class LearnablePositionEmbedder2D(nn.Module):
    """2D可学习位置嵌入"""
    def __init__(self, size: Union[int, tuple], dim: int):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.dim = dim
        self.embed = nn.Parameter(torch.randn(1, dim, self.size[0], self.size[1]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.size[0] and W == self.size[1], "Size mismatch"
        return x + self.embed.expand(B, -1, -1, -1)

class LearnablePositionEmbedder1D(nn.Module):
    """1D可学习位置嵌入"""
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.size = size
        self.dim = dim
        self.embed = nn.Parameter(torch.randn(1, dim, size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape  # 假设输入为 (B, C, H*W) 展平后的1D序列
        assert L == self.size, "Size mismatch"
        return x + self.embed.expand(B, -1, -1)

class FeedForwardNetwork(nn.Module):
    """前馈网络（原parallel_decoder中的子模块）"""
    def __init__(self, dim: int, dim_hidden: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.net = Sequential(
            Linear(dim, dim_hidden),
            nn.ReLU(inplace=True),
            Linear(dim_hidden, dim),
            Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """多头注意力（原parallel_decoder中的子模块）"""
    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, 
        dim_key: Optional[int] = None, dim_value: Optional[int] = None
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        
        dim_key = dim_key or dim
        dim_value = dim_value or dim
        
        self.q_proj = Linear(dim, dim, bias=True)
        self.k_proj = Linear(dim_key, dim, bias=True)
        self.v_proj = Linear(dim_value, dim, bias=True)
        self.out_proj = Linear(dim, dim, bias=True)
        self.dropout = Dropout(dropout)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        memory = x if memory is None else memory
        Q = self.q_proj(x).view(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        K = self.k_proj(memory).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = self.v_proj(memory).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ V).transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(x)

class ParallelDecoder(nn.Module):
    """并行解码器（原parallel_decoder中的核心模块）"""
    def __init__(
        self, dim: int, num_heads: int, 
        dim_ffn: int = 2048, dropout: float = 0.0, dim_memory: Optional[int] = None
    ):
        super().__init__()
        #self.norm1 = BatchNorm1d(dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = BatchNorm1d(dim)
        self.ffn1 = FeedForwardNetwork(dim, dim_ffn, dropout)
        self.ffn2 = FeedForwardNetwork(dim, dim_ffn, dropout)
        self.cross_attn = MultiHeadAttention(
            dim, num_heads, dropout, dim_key=dim_memory, dim_value=dim_memory
        )
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.ffn1(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), memory=memory)
        x = x + self.ffn2(self.norm3(x))
        return x

class CrossAttention(nn.Module):
    """交叉注意力去噪器（原denoiser中的实现）"""
    def __init__(
        self, pool: nn.Module, embedder: Optional[nn.Module], 
        norm: Optional[nn.Module], decoder: ParallelDecoder
    ):
        super().__init__()
        self.pool = pool
        self.embedder = embedder
        self.norm = norm or Identity()
        self.decoder = decoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.pool(x)  # (B, C, H', W')
        if self.embedder:
            x = self.embedder(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H'W', C)
        x = self.norm(x)
        x = self.decoder(x, memory=x)  # 自注意力解码
        x = x.mean(dim=1)  # 全局平均池化得到特征向量
        return x

class LinearProjection(nn.Module):
    """线性投影去噪器"""
    def __init__(self, dim: int, memory_channels: int):
        super().__init__()
        self.projector = Sequential(
            Linear(memory_channels, dim, bias=True),
            BatchNorm1d(dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).mean(dim=2)  # 全局平均池化 (B, C)
        return self.projector(x)

# --------------------------- 核心FMN模块 --------------------------- #

@dataclass
class FMNConfig:
    """配置类（替代原cfg对象）"""
    extractor_type: str = "TripleConvHead"    # {"TripleConvHead", "BackboneHead", "Identity"}
    denoiser_type: str = "CrossAttention"     # {"CrossAttention", "LinearProjection"}
    
    # CrossAttention专属配置
    pool_type: str = "Max"                    # {"Max", "Avg"}
    pool_size: Union[int, tuple] = (7, 7)
    pos_emb_type: str = "Learnable2D"         # {"Learnable2D", "Learnable1D", None}
    layer_norm: bool = True
    pd_num_heads: int = 8
    pd_dim_ffn: int = 2048
    pd_dropout: float = 0.0
    
    # 公共参数
    input_channels: int = 2048  # Backbone输出通道数
    embed_dim: int = 1024       # 特征嵌入维度（与BMN对齐）

class TripleConvHead(nn.Module):
    """三层卷积头提取器（原实现）"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.in_channels = in_channels
        self.out_channels = in_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class ForegroundModulationNetwork(nn.Module):
    """完全解耦的前景调制网络"""
    def __init__(self, config: FMNConfig):
        super().__init__()
        self.config = config
        self.extractor = self._build_extractor()
        self.denoiser = self._build_denoiser()
    
    def _build_extractor(self) -> nn.Module:
        if self.config.extractor_type == "TripleConvHead":
            return TripleConvHead(self.config.input_channels)
        elif self.config.extractor_type == "BackboneHead":
            # 若需支持BackboneHead，需外部传入BackboneHead实现，此处示例用Identity代替
            return Identity()
        elif self.config.extractor_type == "Identity":
            return Identity()
        else:
            raise ValueError(f"未知提取器类型: {self.config.extractor_type}")
    
    def _build_denoiser(self) -> nn.Module:
        if self.config.denoiser_type == "CrossAttention":
            pool = nn.AdaptiveMaxPool2d(self.config.pool_size) if self.config.pool_type == "Max" else \
                   nn.AdaptiveAvgPool2d(self.config.pool_size)
            
            embedder = None
            if self.config.pos_emb_type == "Learnable2D":
                embedder = LearnablePositionEmbedder2D(self.config.pool_size, self.config.input_channels)
            elif self.config.pos_emb_type == "Learnable1D":
                embedder = LearnablePositionEmbedder1D(self.config.pool_size[0], self.config.input_channels)
            
            norm = LayerNorm(self.config.input_channels) if self.config.layer_norm else Identity()
            decoder = ParallelDecoder(
                dim=self.config.embed_dim,
                num_heads=self.config.pd_num_heads,
                dim_ffn=self.config.pd_dim_ffn,
                dropout=self.config.pd_dropout,
                dim_memory=self.config.input_channels
            )
            return CrossAttention(pool, embedder, norm, decoder)
        
        elif self.config.denoiser_type == "LinearProjection":
            return LinearProjection(
                dim=self.config.embed_dim,
                memory_channels=self.config.input_channels
            )
        else:
            raise ValueError(f"未知去噪器类型: {self.config.denoiser_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone输出的特征图 (B, C, H, W)
        Returns:
            去噪后的特征向量 (B, embed_dim)
        """
        x = F.normalize(x, p=2, dim=1)  # 新增L2归一化
        x = self.extractor(x)
        return self.denoiser(x)

# --------------------------- 使用示例 --------------------------- #

if __name__ == "__main__":
    # 初始化配置（替代原cfg.MODEL.REID相关参数）
    config = FMNConfig(
        input_channels=2048,  # 假设Backbone输出通道为2048（如ResNet50的layer4输出）
        embed_dim=1024,        # 与原代码中DIM_IDENTITY一致
    )
    fmn = ForegroundModulationNetwork(config)
    
    # 模拟输入特征图 (Batch=2, Channel=2048, Height=56, Width=56)
    input_feat = torch.randn(2, 2048, 56, 56)
    output_feat = fmn(input_feat)
    print(f"输出特征形状: {output_feat.shape}")  # 应输出 (2, 1024)