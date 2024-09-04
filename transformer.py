# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Transformer model."""

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # Vocabulary size.
  vocab_size: int
  # The dimension of the first embedding.
  embedding_dim: int = 64
  # The number of multi-head attention layers.
  num_layers: int = 4
  # The number of heads per layer.
  num_heads: int = 8
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4



class MultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention (Vaswani et al., 2017)."""

    def __init__(
        self,
        num_heads: int,
        num_hiddens_per_head: int
    ) -> None:
        """Initializes the attention module.

        Args:
          num_heads: Number of heads to use.
          num_hiddens_per_head: Number of hidden neurons per head.
          name: Name of the module.
        """
        super(MultiHeadDotProductAttention, self).__init__()
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        self.num_hiddens = num_hiddens_per_head * num_heads
        
        self.q_linear = nn.Linear(self.num_hiddens, self.num_hiddens, bias=False)
        self.k_linear = nn.Linear(self.num_hiddens, self.num_hiddens, bias=False)
        self.v_linear = nn.Linear(self.num_hiddens, self.num_hiddens, bias=False)
        self.output_linear = nn.Linear(self.num_hiddens, num_hiddens_per_head * num_heads, bias=False)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the output of the multi-head attention."""
        batch_size, sequence_length, embedding_size = inputs_q.size()

        # Apply linear layers
        q = self.q_linear(inputs_q)  # Shape [B, T, num_hiddens]
        k = self.k_linear(inputs_kv) # Shape [B, T, num_hiddens]
        v = self.v_linear(inputs_kv) # Shape [B, T, num_hiddens]

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self._num_heads, self._num_hiddens_per_head).transpose(1, 2)
        k = k.view(batch_size, -1, self._num_heads, self._num_hiddens_per_head).transpose(1, 2)
        v = v.view(batch_size, -1, self._num_heads, self._num_hiddens_per_head).transpose(1, 2)


        # Scaled dot-product attention
        attention = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self._num_hiddens_per_head, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        
        # Softmax normalization
        normalized_attention = F.softmax(attention, dim=-1)
        # Attention output
        output = torch.matmul(normalized_attention, v)  # Shape [B, num_heads, T, num_hiddens_per_head]
        
        # Reshape to concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.num_hiddens)
        
        # Final linear transformation
        output = self.output_linear(output)
        return output

def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> torch.Tensor:
    """Creates sinusoidal encodings from the original transformer paper.

    The returned values are, for all i < D/2:
      array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
      array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

    Args:
      sequence_length: Sequence length.
      hidden_size: Dimension of the positional encoding vectors, D. Should be
        even.
      max_timescale: Maximum timescale for the frequency.

    Returns:
      A tensor of shape [L, D].
    """
    freqs = np.arange(0, hidden_size, 2)
    inv_freq = max_timescale ** (-freqs / hidden_size)

    pos_seq = np.arange(start=0, stop=sequence_length)

    sinusoid_inp = np.outer(pos_seq, inv_freq)
    embeddings = np.concatenate(
        [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
    )
    return torch.tensor(embeddings[:, :hidden_size], dtype=torch.float32)


def embed_sequences(sequences: torch.Tensor, config: TransformerConfig) -> torch.Tensor:
    """Returns embeddings for sequences of tokens."""
    # Initialize the embedding layer with truncated normal distribution
    embs_init = torch.nn.init.trunc_normal_
    embeddings_layer = nn.Embedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.embedding_dim
    )
    embeddings_layer.weight.data = embs_init(embeddings_layer.weight.data, std=config.emb_init_scale)
    
    embeddings = embeddings_layer(sequences)
    embeddings *= torch.sqrt(torch.tensor(config.embedding_dim, dtype=torch.float32))

    _, sequence_length, embedding_size = embeddings.size()
    pos_encodings = sinusoid_position_encoding(
        sequence_length=sequence_length,
        hidden_size=embedding_size,
    ).to(embeddings.device)
    return embeddings + pos_encodings


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    """Helper function for layer norm."""
    return nn.LayerNorm(normalized_shape=x.size()[-1], elementwise_affine=True)(x)

def shift_right(sequences: torch.Tensor) -> torch.Tensor:
    """Right-shift the input by padding on the temporal axis."""
    bos_array = torch.zeros((sequences.size(0), 1), dtype=torch.long, device=sequences.device)
    padded_sequences = torch.cat([bos_array, sequences], dim=1)
    return padded_sequences[:, :-1]


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': MultiHeadDotProductAttention(
                    num_heads=config.num_heads,
                    num_hiddens_per_head=config.embedding_dim // config.num_heads
                ),
                'linear1': nn.Linear(config.embedding_dim, config.embedding_dim * config.widening_factor),
                'linear2': nn.Linear(config.embedding_dim * config.widening_factor, config.embedding_dim),
            })
            for _ in range(config.num_layers)
        ])
        self.final_linear = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        """Returns the transformer decoder output, shape [B, T, V]."""
        # Right shift the targets to get the inputs (the first token is now a 0).
        inputs = shift_right(targets)

        # Embeds the inputs and adds positional encodings.
        embeddings = embed_sequences(inputs, self.config)

        batch_size, sequence_length = embeddings.size()[:2]

        # The causal mask is shared across heads.
        causal_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), device=embeddings.device)
        ).unsqueeze(0).unsqueeze(0)  # shape [1, 1, T, T]

        h = embeddings
        for layer in self.layers:
            self_attention = layer['self_attention'](inputs_q=h, inputs_kv=h, mask=causal_mask)
            attention = layer_norm(h + self_attention)

            # Position-wise feedforward network.
            h = F.gelu(layer['linear1'](attention))
            h = layer_norm(layer['linear2'](h) + attention)

        logits = self.final_linear(h)
        return F.log_softmax(logits, dim=-1)
