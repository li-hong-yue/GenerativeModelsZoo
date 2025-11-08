from pathlib import Path

from .vqgan_tokenizer import VQGANTokenizer
from .vqgan_transformer import VQGANTransformer

class VQGAN:
    def __init__(self, **kwargs):
        mode = kwargs.pop('mode', None)
        
        if mode not in ['tokenizer', 'transformer']: # auto decide mode
            checkpoint_dir = Path(kwargs['checkpoint_dir'])
            tokenizer_ckpt = checkpoint_dir / 'tokenizer_checkpoint_latest.pt'
            transformer_ckpt = checkpoint_dir / 'transformer_checkpoint_latest.pt'

            if tokenizer_ckpt.exists():
                mode = 'transformer'
            else:
                # Default if no checkpoint exists
                mode = 'tokenizer'

        # Initialize the chosen model
        if mode == 'tokenizer':
            self.model = VQGANTokenizer(**kwargs)
        elif mode == 'transformer':
            self.model = VQGANTransformer(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        print(f"initializing VQGAN_{mode}")

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying model.
        This allows calling methods like encode, decode, or forward transparently.
        """
        return getattr(self.model, name)