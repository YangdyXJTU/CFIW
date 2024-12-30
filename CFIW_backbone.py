import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Block as TimmBlock
from mamba_ssm_2 import Mamba
    
class ResidualMamba(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Save the input for residual connection
        identity = x
        # Apply LayerNorm and Mamba
        out = self.norm(x)
        out = self.mamba(out)
        # Add residual connection
        return out + identity

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_conv_stem=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        
        # Choose to use ConvStem or PatchEmbed according to the parameters
        if use_conv_stem:
            self.patch_embed = RDCN_Stem(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans,
                embed_dim=embed_dim)
        else:
            self.patch_embed = RDCN_Stem(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans,
                embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ResidualMamba(
                d_model=embed_dim,  # If it is the dimension after concatenating frequency and spatial features
            ) 
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
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
        B = x.shape[0]
        x = self.patch_embed(x)
        # Add cls_token to each position instead of concatenating
        cls_tokens = self.cls_token.expand(B, x.shape[1], -1)
        x = x + self.cls_token
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x    # return all features 

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
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
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        
        # Image domain branch only uses one layer of transformer block
        self.image_domain_branch = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=1,  # Only use one layer
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_conv_stem=False
        )
        
        # Waveform domain branch only uses one layer of transformer block
        self.waveform_domain_branch = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=15,
            embed_dim=embed_dim,
            depth=1,  # Only use one layer
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_conv_stem=True
        )

        
        # Deep feature learning after fusion
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth-4)]  # depth-2 is because 2 layers have been used before        
        # Replace TimmBlock with MambaBlock
        self.post_blocks = nn.ModuleList([
            ResidualMamba(
                d_model=embed_dim
            ) for _ in range(depth-2)
        ])
        
        
        self.norm = nn.LayerNorm(embed_dim)
        # ... existing code ...
        self.pool = nn.AdaptiveMaxPool1d(1)  # Add pooling layer
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.fusion = CrossAttentionFusion(embed_dim, num_heads)

    def optical_flow_transform(self, x):
        B, C, H, W = x.shape
        
        # Convert to grayscale image
        if C == 3:
            x_gray = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
        else:
            x_gray = x
        
        # Gaussian smoothing
        kernel_size = 5
        sigma = 1.5
        gaussian_kernel = torch.exp(torch.tensor([[(i-kernel_size//2)**2 + (j-kernel_size//2)**2 for j in range(kernel_size)] for i in range(kernel_size)]) / (-2*sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1).to(x.device)
        x_smooth = F.conv2d(x_gray, gaussian_kernel, padding=kernel_size//2)
        
        # Calculate multiscale gradients
        features = []
        current_x = x_smooth
        
        for scale in [1, 2, 4]:
            # Calculate the gradient of the current scale
            dx = F.conv2d(current_x, 
                         torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3).repeat(1, 1, 1, 1).to(x.device),
                         padding=(0, 1))
            
            dy = F.conv2d(current_x,
                         torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1).repeat(1, 1, 1, 1).to(x.device),
                         padding=(1, 0))
            
            # Calculate more optical flow features
            magnitude = torch.sqrt(dx**2 + dy**2)  # Gradient magnitude
            angle = torch.atan2(dy, dx)  # Gradient direction
            
            # Calculate Laplacian features
            laplacian = F.conv2d(current_x,
                                torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(x.device),
                                padding=1)
            
            # Calculate second order derivatives
            dxx = F.conv2d(dx, 
                          torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3).repeat(1, 1, 1, 1).to(x.device),
                          padding=(0, 1))
            dyy = F.conv2d(dy,
                          torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1).repeat(1, 1, 1, 1).to(x.device),
                          padding=(1, 0))
            
            # Add all features to the list
            if scale > 1:
                magnitude = F.interpolate(magnitude, size=(H, W), mode='bilinear', align_corners=False)
                angle = F.interpolate(angle, size=(H, W), mode='bilinear', align_corners=False)
                laplacian = F.interpolate(laplacian, size=(H, W), mode='bilinear', align_corners=False)
                dxx = F.interpolate(dxx, size=(H, W), mode='bilinear', align_corners=False)
                dyy = F.interpolate(dyy, size=(H, W), mode='bilinear', align_corners=False)
            
            features.extend([magnitude, angle, laplacian, dxx, dyy])
            
            # Downsample for the next scale
            if scale < 4:
                current_x = F.avg_pool2d(current_x, kernel_size=2, stride=2)
        
        # Merge all features
        flow = torch.cat(features, dim=1)
        
        # Normalize
        flow = (flow - flow.mean(dim=[2,3], keepdim=True)) / (flow.std(dim=[2,3], keepdim=True) + 1e-6)
        
        return flow

    def forward(self, x):
        # Image feature extraction (one layer)
        image_feat = self.image_domain_branch.forward_features(x)

        # Waveform feature extraction (one layer)
        x_wave = self.optical_flow_transform(x)
        
        waveform_feat = self.waveform_domain_branch.forward_features(x_wave)

        fused_feat = self.fusion(image_feat, waveform_feat)  # feature fusion of cross-attention
        # fused_feat = image_feat + waveform_feat   # simple feature fusion
        # fused_feat = image_feat       # only image feature
        # fused_feat = waveform_feat       # only waveform feature
        
        # Use remained mamba blocks learning deep features
        for blk in self.post_blocks:
            fused_feat = blk(fused_feat)
        
        # Modify this part of the code
        fused_feat = self.norm(fused_feat)
        fused_feat = fused_feat.transpose(1, 2)  # From [B, N, C] to [B, C, N]
        pooled_feat = self.pool(fused_feat)  # [B, C, 1]
        pooled_feat = pooled_feat.transpose(1, 2)  # [B, 1, C]
        pooled_feat = pooled_feat.squeeze(1)  # [B, C]
        out = self.head(pooled_feat)
        
        return out

class RDCN_Stem(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[0])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.num_levels = 4  # Corresponding to the original 4 levels
        self.channels = [
            embed_dim // 8,   # First level
            embed_dim // 4,   # Second level
            embed_dim // 2,   # Third level
            embed_dim        # Last level
        ]
 
        self.tcn_blocks = nn.ModuleList()
        for i in range(self.num_levels):
            in_ch = in_chans if i == 0 else self.channels[i-1]
            out_ch = self.channels[i]
            dilation = 2 ** i  # Exponentially growing dilation rate
            
            self.tcn_blocks.append(
                TemporalBlock(
                    in_ch, 
                    out_ch,
                    kernel_size=3,
                    stride=2,  # Downsample every layer
                    dilation=dilation,
                    padding=dilation,
                    dropout=0.1
                )
            )

        # Final smoothing layer
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

        # Store the features of each layer
        features = []
        
        # Forward through TCN blocks
        current = x
        for i, block in enumerate(self.tcn_blocks):
            current = block(current)
            features.append(current)

        # Feature pyramid fusion (from bottom to top)
        p4 = features[-1]  # Already embed_dim channels
        # Final smoothing
        out = self.smooth(p4)
        
        if self.flatten:
            out = out.flatten(2).transpose(1, 2)
        
        out = self.norm(out)
        return out

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        # Main branch
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
        
        # Activation and dropout
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        if stride != 1 or n_inputs != n_outputs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_inputs, n_outputs, 1, stride=stride),
                nn.BatchNorm2d(n_outputs)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim  # Add this line to store dim value
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        #  to frequency attention
        self.q_image_domain = nn.Linear(dim, dim)
        self.kv_wave_domain = nn.Linear(dim, dim * 2)
        
        # Frequency to spatial attention
        self.q_wave_domain = nn.Linear(dim, dim)
        self.kv_image_domain = nn.Linear(dim, dim * 2)
        
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, image_domain_feat, wave_domain_feat):
        B = image_domain_feat.shape[0]

        q_i = self.q_image_domain(image_domain_feat).reshape(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        kv_w = self.kv_wave_domain(wave_domain_feat).reshape(B, -1, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k_w, v_w = kv_w[0], kv_w[1]
        
        attn_i2w = (q_i @ k_w.transpose(-2, -1)) * self.scale
        # Add numerical stability
        attn_i2w = torch.clamp(attn_i2w, min=-100, max=100)
        attn_i2w = attn_i2w.softmax(dim=-1)
        image_domain_attended = (attn_i2w @ v_w).transpose(1, 2).reshape(B, -1, self.dim)
        
        q_w = self.q_wave_domain(wave_domain_feat).reshape(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        kv_i = self.kv_image_domain(image_domain_feat).reshape(B, -1, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k_i, v_i = kv_i[0], kv_i[1]
        
        attn_w2i = (q_w @ k_i.transpose(-2, -1)) * self.scale
        attn_w2i = attn_w2i.softmax(dim=-1)
        # Ensure output dimension is correct
        wave_domain_attended = (attn_w2i @ v_i).transpose(1, 2).reshape(B, -1, self.dim)
        
        # Fusion and projection
        fused = torch.cat([image_domain_attended, wave_domain_attended], dim=-1)
        # Ensure output dimension is correct
        out = self.proj(fused)
        out = self.norm(out)
        # if out.dim() > 2:
        #     out = out.squeeze()  # Remove all dimensions with size 1
        return out