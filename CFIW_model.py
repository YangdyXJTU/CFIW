from timm.models.vision_transformer import _cfg
from CFIW_backbone import CFIW_net

__all__ = ['CFIW']

def CFIW(**kwargs):

    model = CFIW_net(
                        img_size=224,
                        patch_size=16, 
                        in_chans=3,
                        num_classes=5,
                        embed_dim=384,
                        depth=8,
                        num_heads=12,
                        )
    model.default_cfg = _cfg()
    return model
