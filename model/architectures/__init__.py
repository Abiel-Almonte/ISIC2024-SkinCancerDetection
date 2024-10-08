from .unet_tabnet import *
from .transformer import *
from .vae_tabnet import *
from .lightnn import *
from .base import *

__all__ = []
__all__.extend(unet_tabnet.__all__)
__all__.extend(transformer.__all__)
__all__.extend(vae_tabnet.__all__)
__all__.extend(base.__all__)

from .registry import *

register_model('EfficientUNetWithTabular', EfficientUNetWithTabular, {'cont_features': 31, 'bin_features': 6})
register_model('EfficientUNetWithTabular_v2', EfficientUNetWithTabular_v2, {'cont_features': 31, 'bin_features': 6})
register_model('ViTWithTabular', ViTWithTabular, {'cont_features': 31, 'bin_features': 6})
register_model('VAETabNet', VAETabNet, {'cont_features': 31, 'bin_features': 6, 'latent_dim': 256})
register_model('CrossModalTransformer', CrossModalTransformer, {'cont_features': 31, 'bin_features': 6, 'image_size': 224, 'dim':6, 'depth':768, 'mlp_dim': 3072})
register_model('LightMultiModalNN', LightMultiModalNN, {'tabular_hidden_dim': 128, 'tabular_out_dim': 384, 'tabular_dropout': 0.3, 'classifier_hidden_dim':64, 'classifier_droupout': 0.3})

__all__.extend(registry.__all__)