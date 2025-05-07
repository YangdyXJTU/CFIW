import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm_2 import Mamba


class ResidualMamba(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, mlp_ratio=4.0, drop=0., drop_path=0.3):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        norm_layer = nn.LayerNorm
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        
        mlp_hidden_dim = int(d_model * mlp_ratio)
        
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop
        )
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        mamba_out = self.drop_path1(self.mamba(self.norm1(x)))
        mlp_out = self.drop_path2(self.mlp(self.norm2(x)))
        x = x + mamba_out + mlp_out
        return x


class RDCN_stem(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5,
                 embed_dim=768, depth=12, drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        
        self.patch_embed = RDCN_block(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            ResidualMamba(
                d_model=embed_dim,
            ) 
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.cls_token
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


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


class CFIW_net(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16, 
                 in_chans=3,
                 num_classes=5,
                 embed_dim=384,
                 depth=6,
                 num_heads=12):
        super().__init__()
        
        self.image_domain_branch = RDCN_stem(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=1
        )
        
        self.waveform_domain_branch = RDCN_stem(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=15,
            embed_dim=embed_dim,
            depth=1
        )

        self.post_blocks = nn.ModuleList([
            ResidualMamba(
                d_model = embed_dim,
            ) for _ in range(depth-2)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.fusion = CrossAttentionFusion(embed_dim, num_heads)

    def optical_flow_transform(self, x):
        B, C, H, W = x.shape
        
        if C == 3:
            x_gray = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
        else:
            x_gray = x

        kernel_size = 3
        sigma = 1.5
        
        kernel_1d = torch.exp(-torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1)**2 / (2*sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_v = kernel_1d.view(1, 1, -1, 1).to(x.device)
        kernel_h = kernel_1d.view(1, 1, 1, -1).to(x.device)
        
        x_smooth = F.conv2d(x_gray, kernel_v, padding=(kernel_size//2, 0))
        x_smooth = F.conv2d(x_smooth, kernel_h, padding=(0, kernel_size//2))
        
        features = []
        current_x = x_smooth
        
        for scale in [1, 2, 4]:
            dx = F.conv2d(current_x, 
                         torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3).repeat(1, 1, 1, 1).to(x.device),
                         padding=(0, 1))
            
            dy = F.conv2d(current_x,
                         torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1).repeat(1, 1, 1, 1).to(x.device),
                         padding=(1, 0))
            
            magnitude = torch.sqrt(dx**2 + dy**2)
            angle = torch.atan2(dy, dx)
            
            laplacian = F.conv2d(current_x,
                                torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(x.device),
                                padding=1)
            
            dxx = F.conv2d(dx, 
                          torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3).repeat(1, 1, 1, 1).to(x.device),
                          padding=(0, 1))
            dyy = F.conv2d(dy,
                          torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1).repeat(1, 1, 1, 1).to(x.device),
                          padding=(1, 0))
            
            if scale > 1:
                magnitude = F.interpolate(magnitude, size=(H, W), mode='bilinear', align_corners=False)
                angle = F.interpolate(angle, size=(H, W), mode='bilinear', align_corners=False)
                laplacian = F.interpolate(laplacian, size=(H, W), mode='bilinear', align_corners=False)
                dxx = F.interpolate(dxx, size=(H, W), mode='bilinear', align_corners=False)
                dyy = F.interpolate(dyy, size=(H, W), mode='bilinear', align_corners=False)
            
            features.extend([magnitude, angle, laplacian, dxx, dyy])
            
            if scale < 4:
                current_x = F.avg_pool2d(current_x, kernel_size=2, stride=2)
        
        flow = torch.cat(features, dim=1)
        flow = (flow - flow.mean(dim=[2,3], keepdim=True)) / (flow.std(dim=[2,3], keepdim=True) + 1e-6)
        
        return flow

    def forward(self, x):
        image_domain_feat = self.image_domain_branch.forward_features(x)

        with torch.no_grad():
            x_waveform = self.optical_flow_transform(x)
        
        waveform_domain_feat = self.waveform_domain_branch.forward_features(x_waveform)

        fused_feat = self.fusion(image_domain_feat, waveform_domain_feat)
        
        for blk in self.post_blocks:
            fused_feat = blk(fused_feat)
        
        fused_feat = self.norm(fused_feat)
        fused_feat = fused_feat.transpose(1, 2)
        pooled_feat = self.pool(fused_feat)
        pooled_feat = pooled_feat.transpose(1, 2)
        pooled_feat = pooled_feat.squeeze(1)
        out = self.head(pooled_feat)
        
        return out


class RDCN_block(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.num_levels = 4
        self.channels = [
            embed_dim // 8,
            embed_dim // 4,
            embed_dim // 2,
            embed_dim
        ]
        
        self.tcn_blocks = nn.ModuleList()
        for i in range(self.num_levels):
            in_ch = in_chans if i == 0 else self.channels[i-1]
            out_ch = self.channels[i]
            dilation = 2 ** i
            
            self.tcn_blocks.append(
                TemporalBlock(
                    in_ch, 
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    dilation=dilation,
                    padding=dilation,
                    dropout=0.1
                )
            )

        self.smooth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        features = []
        
        current = x
        for i, block in enumerate(self.tcn_blocks):
            current = block(current)
            features.append(current)

        p4 = features[-1]
        out = self.smooth(p4)
        
        if self.flatten:
            out = out.flatten(2).transpose(1, 2)
        
        out = self.norm(out)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.conv2 = nn.Conv2d(
            n_outputs, n_outputs, kernel_size,
            stride=1, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(n_outputs)
        
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        if stride != 1 or n_inputs != n_outputs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_inputs, n_outputs, 1, stride=stride),
                nn.BatchNorm2d(n_outputs)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_image_domain = nn.Linear(dim, dim)
        self.kv_waveform_domain = nn.Linear(dim, dim * 2)
        
        self.q_waveform_domain = nn.Linear(dim, dim)
        self.kv_image_domain = nn.Linear(dim, dim * 2)
        
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, image_domain_feat, waveform_domain_feat):
        B = image_domain_feat.shape[0]

        q_s = self.q_image_domain(image_domain_feat).reshape(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        kv_f = self.kv_waveform_domain(waveform_domain_feat).reshape(B, -1, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k_f, v_f = kv_f[0], kv_f[1]
        
        attn_s2f = (q_s @ k_f.transpose(-2, -1)) * self.scale
        attn_s2f = torch.clamp(attn_s2f, min=-100, max=100)
        attn_s2f = attn_s2f.softmax(dim=-1)
        image_domain_attended = (attn_s2f @ v_f).transpose(1, 2).reshape(B, -1, self.dim)
        
        q_f = self.q_waveform_domain(waveform_domain_feat).reshape(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        kv_s = self.kv_image_domain(image_domain_feat).reshape(B, -1, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k_s, v_s = kv_s[0], kv_s[1]
        
        attn_f2s = (q_f @ k_s.transpose(-2, -1)) * self.scale
        attn_f2s = attn_f2s.softmax(dim=-1)
        waveform_domain_attended = (attn_f2s @ v_s).transpose(1, 2).reshape(B, -1, self.dim)
        
        fused = torch.cat([image_domain_attended, waveform_domain_attended], dim=-1)
        out = self.proj(fused)
        out = self.norm(out)
        
        return out