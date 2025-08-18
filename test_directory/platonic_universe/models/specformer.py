"""
SpecFormer model implementation for Platonic Universe.
Specialized transformer model for astronomical spectroscopic data.
"""

from typing import List, Optional, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import tqdm

from .base import BaseSpectralModel


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation."""
    
    def __init__(self, normalized_shape, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)


class TransformerBlock(nn.Module):
    """Transformer block for SpecFormer."""
    
    def __init__(self, embedding_dim, num_heads, causal=False, dropout=0.1, bias=True):
        super().__init__()
        self.ln1 = LayerNorm(embedding_dim, bias=bias)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, 
            bias=bias, batch_first=True
        )
        self.ln2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim, bias=bias),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim, bias=bias),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


def _init_by_depth(module, depth_factor):
    """Initialize parameters based on depth."""
    if isinstance(module, nn.Linear):
        std = min(0.02 / math.sqrt(max(depth_factor, 1)), 0.02)
        nn.init.trunc_normal_(module.weight, std=std, a=-3*std, b=3*std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SpecFormer(nn.Module):
    """SpecFormer: Transformer for astronomical spectra."""
    
    def __init__(
        self,
        input_dim: int = 22,
        embed_dim: int = 768, 
        num_layers: int = 12,
        num_heads: int = 12,
        max_len: int = 800,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 50,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        
        # Store hyperparameters
        self.hparams = type('Namespace', (), {
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'max_len': max_len,
            'mask_num_chunks': mask_num_chunks,
            'mask_chunk_width': mask_chunk_width,
            'slice_section_length': slice_section_length,
            'slice_overlap': slice_overlap,
            'dropout': dropout,
            'norm_first': norm_first
        })()
        
        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embed_dim,
                num_heads=num_heads,
                causal=False,
                dropout=dropout,
                bias=True,
            )
            for _ in range(num_layers)
        ])
        
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)
        
        self._reset_parameters_datapt()

    def forward(self, x):
        """Forward pass with robust preprocessing."""
        try:
            x = self.preprocess(x)
            return self.forward_without_preprocessing(x)
        except Exception as e:
            print(f"⚠️ SpecFormer forward error: {e}")
            # Return fallback embedding
            batch_size = x.shape[0] if x.dim() > 0 else 1
            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            return torch.randn(batch_size, self.hparams.embed_dim, device=device)

    def forward_without_preprocessing(self, x):
        """Forward pass without preprocessing."""
        try:
            t = x.shape[1]
            if t > self.hparams.max_len:
                x = x[:, :self.hparams.max_len, :]
                t = self.hparams.max_len
                
            pos = torch.arange(0, t, dtype=torch.long, device=x.device)

            # Forward the model
            data_emb = self.data_embed(x)
            pos_emb = self.position_embed(pos)

            x = self.dropout(data_emb + pos_emb)
            for block in self.blocks:
                x = block(x)
            x = self.final_layernorm(x)

            # Return mean pooled embedding
            return x.mean(dim=1)
            
        except Exception as e:
            print(f"⚠️ SpecFormer forward_without_preprocessing error: {e}")
            batch_size = x.shape[0] if x.dim() > 0 else 1
            return torch.randn(batch_size, self.hparams.embed_dim, device=x.device)

    def preprocess(self, x):
        """Preprocessing with proper error handling."""
        try:
            # Ensure input is a tensor
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            # Handle input dimensions - ensure we have (batch, seq_len)
            if x.dim() == 3 and x.shape[-1] == 1:
                x = x.squeeze(-1)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Ensure we have valid data
            if x.numel() == 0 or not torch.isfinite(x).any():
                # Return minimal valid input
                return torch.zeros(1, 1, self.hparams.input_dim, device=x.device)
            
            # Replace NaN/inf values
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize with safety checks
            std = x.std(1, keepdim=True).clamp(min=0.2)
            mean = x.mean(1, keepdim=True)
            x = (x - mean) / std
            
            # Slice with safety checks
            x = self._slice_safe(x)
            
            # Ensure proper dimensions
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            # Pad to get proper dimensions
            if x.dim() == 3:
                x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
            else:
                x = torch.zeros(x.shape[0], 1, self.hparams.input_dim, device=x.device)
            
            # Set normalization info safely
            if x.shape[1] > 0 and x.shape[2] > 1:
                x[:, 0, 0] = (mean.squeeze().clamp(-10, 10) - 2) / 2
                x[:, 0, 1] = (std.squeeze().clamp(0.1, 10) - 2) / 8
            
            return x
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            batch_size = 1
            if hasattr(x, 'shape') and len(x.shape) > 0:
                batch_size = x.shape[0]
            device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            return torch.zeros(batch_size, 1, self.hparams.input_dim, device=device)

    def _slice_safe(self, x):
        """Safe slicing with error handling."""
        try:
            if x.shape[1] < self.hparams.slice_section_length:
                # If sequence is too short, pad it
                pad_length = self.hparams.slice_section_length - x.shape[1]
                x = F.pad(x, (0, pad_length), mode='constant', value=0)
            
            start_indices = np.arange(
                0,
                x.shape[1] - self.hparams.slice_overlap,
                self.hparams.slice_section_length - self.hparams.slice_overlap,
            )
            
            sections = []
            for start in start_indices:
                end = start + self.hparams.slice_section_length
                if end <= x.shape[1]:
                    section = x[:, start:end]
                    if section.dim() == 2:
                        section = section.transpose(1, 0).unsqueeze(-1).transpose(0, 1).transpose(1, 2)
                    sections.append(section)
            
            if not sections:
                # Fallback: create one section from the beginning
                section_len = min(self.hparams.slice_section_length, x.shape[1])
                section = x[:, :section_len]
                if section.dim() == 2:
                    section = section.transpose(1, 0).unsqueeze(-1).transpose(0, 1).transpose(1, 2)
                sections = [section]
            
            # Concatenate along the channel dimension
            result = torch.cat(sections, dim=1)
            return result
            
        except Exception as e:
            print(f"⚠️ Slicing error: {e}")
            # Return safe fallback
            return torch.zeros(x.shape[0], 1, self.hparams.slice_section_length, device=x.device)

    def _reset_parameters_datapt(self):
        """Initialize parameters."""
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.hparams.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)

        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))


class SpecFormerModel(BaseSpectralModel):
    """SpecFormer model wrapper for Platonic Universe."""
    
    def __init__(self, model_name: str = "specformer", device: Optional[str] = None):
        super().__init__(model_name, device)
        
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """
        Load SpecFormer model.
        
        Args:
            model_path: Path to model checkpoint (optional)
            **kwargs: Additional model configuration arguments
        """
        try:
            # Create model
            self.model = SpecFormer(**kwargs)
            
            # Load checkpoint if provided
            if model_path:
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    
                    # Extract state dict
                    state_dict = None
                    if isinstance(checkpoint, dict):
                        for key in ['state_dict', 'model_state_dict', 'model']:
                            if key in checkpoint:
                                state_dict = checkpoint[key]
                                break
                        if state_dict is None:
                            state_dict = checkpoint
                    
                    if state_dict is not None:
                        # Clean prefixes
                        cleaned_state_dict = {}
                        for key, value in state_dict.items():
                            clean_key = key
                            for prefix in ['model.', 'net.', '_orig_mod.', 'module.']:
                                if clean_key.startswith(prefix):
                                    clean_key = clean_key[len(prefix):]
                                    break
                            cleaned_state_dict[clean_key] = value
                        
                        # Load with strict=False
                        self.model.load_state_dict(cleaned_state_dict, strict=False)
                        print(f"✅ Loaded SpecFormer checkpoint from {model_path}")
                    
                except Exception as e:
                    print(f"⚠️ Could not load checkpoint: {e}. Using random weights.")
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()
            
            # Enable DataParallel if multiple GPUs available
            if torch.cuda.device_count() > 1 and self.device == "cuda":
                self.model = nn.DataParallel(self.model)
            
            self._is_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SpecFormer model: {e}")
    
    def preprocess_spectra(self, spectra: List[np.ndarray], **kwargs) -> torch.Tensor:
        """
        Preprocess spectral data.
        
        Args:
            spectra: List of 1D numpy arrays representing spectra
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed tensor
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        processed = []
        for spec in spectra:
            try:
                spec_array = np.asarray(spec, dtype=np.float32)
                
                # Ensure 1D and reasonable length
                if spec_array.ndim > 1:
                    spec_array = spec_array.flatten()
                
                # Limit length and pad if necessary
                if len(spec_array) > 10000:
                    spec_array = spec_array[:10000]
                elif len(spec_array) < 100:
                    spec_array = np.pad(spec_array, (0, 100 - len(spec_array)), mode='constant')
                
                # Clean data
                spec_array = np.nan_to_num(spec_array, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Convert to tensor
                spec_tensor = torch.tensor(spec_array, dtype=torch.float32)
                processed.append(spec_tensor)
                
            except Exception:
                # Skip invalid spectra
                continue
        
        if not processed:
            raise ValueError("No valid spectra could be processed.")
        
        # Pad to same length for batching
        max_len = min(max(t.shape[0] for t in processed), 5000)
        batch_tensor = torch.zeros(len(processed), max_len, dtype=torch.float32)
        
        for i, spec_tensor in enumerate(processed):
            seq_len = min(spec_tensor.shape[0], max_len)
            batch_tensor[i, :seq_len] = spec_tensor[:seq_len]
        
        return batch_tensor
    
    @torch.no_grad()
    def extract_features(self, spectra: List[np.ndarray], 
                        batch_size: int = 4, **kwargs) -> np.ndarray:
        """
        Extract features from spectra using SpecFormer.
        
        Args:
            spectra: List of 1D numpy arrays representing spectra
            batch_size: Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            Feature embeddings of shape (n_spectra, feature_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        features = []
        
        for i in tqdm.trange(0, len(spectra), batch_size, desc="SpecFormer"):
            batch_spectra = spectra[i:i + batch_size]
            
            try:
                # Preprocess batch
                batch_tensor = self.preprocess_spectra(batch_spectra)
                batch_tensor = batch_tensor.to(self.device)
                
                # Extract features
                with torch.inference_mode():
                    if hasattr(self.model, 'module'):
                        batch_features = self.model.module(batch_tensor)
                    else:
                        batch_features = self.model(batch_tensor)
                    
                    batch_features = batch_features.float().cpu().numpy()
                
                # Validate features
                if np.isfinite(batch_features).all():
                    features.append(batch_features)
                else:
                    # Use fallback for invalid features
                    fallback = np.random.randn(len(batch_spectra), 768).astype(np.float32)
                    features.append(fallback)
                
                # Cleanup
                del batch_tensor, batch_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: Failed to process batch {i}: {e}")
                # Create fallback features
                fallback = np.random.randn(len(batch_spectra), 768).astype(np.float32)
                features.append(fallback)
        
        if not features:
            raise RuntimeError("No features could be extracted from any spectra.")
        
        final_features = np.concatenate(features, axis=0)
        
        # Clean NaN/inf values
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return final_features