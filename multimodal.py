import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import math

class CDP_CNNs(nn.Module):
    def __init__(self, in_channels=4, mid_channels=16, out_channels=8):
        super(CDP_CNNs, self).__init__()
        
        self.feature_align = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GELU()
        )

        self.mas_attn = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        self.conv_op_context = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1)),
        )

        self.conv_compress = nn.Conv2d(
            mid_channels * 2, 
            out_channels, 
            kernel_size=1
        )

        self.norm = nn.LayerNorm(out_channels)
        # self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x, num_mas):
        B, J, W, C = x.shape
        assert W % num_mas == 0
        O_max = W // num_mas
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.feature_align(x)
        x = x.view(B, -1, J, O_max, num_mas)

        attn = x.permute(0, 2, 3, 1, 4).contiguous()
        attn = attn.view(-1, x.size(1), 1, num_mas)

        attn = self.mas_attn(attn).view(B, J, O_max, num_mas)
        attn = F.softmax(attn, dim=-1)
        x = (x * attn.unsqueeze(1)).sum(dim=-1)

        x = self.conv_op_context(x) + x
        
        avg_pool = F.adaptive_avg_pool2d(x, (J, 1))
        max_pool = F.adaptive_max_pool2d(x, (J, 1))
        out = torch.cat([avg_pool, max_pool], dim=1)

        out = self.conv_compress(out)
        out = out.squeeze(-1).transpose(1, 2)

        return self.norm(out)


class HTB(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2, feat_drop=0., attn_drop=0.):
        """
        Multi-Head Transformer-style layer for Heterogeneous Machine-Operation interaction.
        """
        super(HTB, self).__init__()
        self.ope_dim = in_dim[0]
        self.mac_dim = in_dim[1]
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads

        # Dropout
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # Multi-head linear projections
        self.W_Q = nn.Linear(self.mac_dim, out_dim, bias=False)
        self.W_K = nn.Linear(self.ope_dim, out_dim, bias=False)
        self.W_V = nn.Linear(self.ope_dim, out_dim, bias=False)
        
        # Standard Transformer practice
        self.W_O = nn.Linear(out_dim, out_dim, bias=False)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(feat_drop),
            nn.Linear(out_dim * 2, out_dim)
        )

        # Residual connection
        if self.mac_dim != out_dim:
            self.res_fc = nn.Linear(self.mac_dim, out_dim, bias=False)
        else:
            self.res_fc = None

        self.norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
        if self.res_fc:
            nn.init.xavier_uniform_(self.res_fc.weight)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, curr_proc_batch, feats):
        op_feats, mac_feats = feats
        B, N, _ = op_feats.shape
        M = mac_feats.shape[1]
        H = self.num_heads
        d_h = self.head_dim

        # Dropout
        op_feats = self.feat_drop(op_feats)
        mac_feats = self.feat_drop(mac_feats)

        # Linear projections
        Q = self.W_Q(mac_feats).view(B, M, H, d_h).transpose(1, 2)
        # [B, N, D] -> [B, N, H, d_h] -> [B, H, N, d_h]
        K = self.W_K(op_feats).view(B, N, H, d_h).transpose(1, 2)
        V = self.W_V(op_feats).view(B, N, H, d_h).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_h ** 0.5)
        mask_ijk = (curr_proc_batch == 1).transpose(1, 2).unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask_ijk, float('-9e10'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, M, self.out_dim)
        attn_output = self.W_O(attn_output)

        # FFN
        mac_res = self.res_fc(mac_feats) if self.res_fc is not None else mac_feats
        out = mac_res + attn_output
        out = self.norm(out + self.ffn(out))

        return out
