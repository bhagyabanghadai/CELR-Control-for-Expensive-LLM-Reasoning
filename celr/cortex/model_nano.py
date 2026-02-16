"""
NanoCortex: A minimal Decision Transformer implementation.
Inspired by Andrej Karpathy's minGPT/nanoGPT, adapted for Offline RL.

Author: AI ML Engineer (CELR Team)
Architecture:
    - Input: Returns-to-Go (scalar), State (vector), Action (discrete)
    - Backbone: GPT-2 style Causal Transformer
    - Output: Next Action Logits
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class NanoConfig:
    state_dim: int = 8          # Dimension of the state vector
    act_dim: int = 7            # Number of discrete actions (CortexAction Enum)
    max_length: int = 20        # Context length (history size) for decision making
    
    # Transformer Hyperparameters (Small/Nano size)
    n_layer: int = 3            # Number of transformer blocks
    n_head: int = 4             # Number of attention heads
    n_embd: int = 128           # Embedding dimension
    dropout: float = 0.1
    vocab_size: int = 1         # Not used for text, but kept for structural similarity if needed.

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including:
    1. Explicit implementation for educational clarity (Karpathy style)
    2. Causal masking
    """

    def __init__(self, config: NanoConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is strictly PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # manual implementation of causal mask
            self.register_buffer("bias", torch.tril(torch.ones(config.max_length * 3, config.max_length * 3))
                                        .view(1, 1, config.max_length * 3, config.max_length * 3))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """ Simple MLP for the FeedForward block """
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoCortex(nn.Module):
    """
    Decision Transformer for CELR.
    Takes (R, s, a) trajectories and predicts the next action 'a'.
    """

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        # Embedding layers
        # 1. State Embedding (Continuous vector -> n_embd)
        self.embed_state = nn.Linear(config.state_dim, config.n_embd)
        # 2. Action Embedding (Discrete index -> n_embd)
        self.embed_action = nn.Embedding(config.act_dim, config.n_embd)
        # 3. Return-to-Go Embedding (Scalar -> n_embd)
        self.embed_return = nn.Linear(1, config.n_embd)
        
        # Positional Embedding
        self.embed_timestep = nn.Embedding(config.max_length, config.n_embd)

        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Action Prediction Head
        self.predict_action = nn.Linear(config.n_embd, config.act_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Weight initialization (Karpathy style)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        Args:
            states: (B, T, state_dim)
            actions: (B, T)
            returns_to_go: (B, T, 1)
            timesteps: (B, T)
        
        Returns:
            action_preds: (B, T, act_dim)
        """
        B, T, _ = states.shape

        # 1. Embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        return_embeddings = self.embed_return(returns_to_go) + time_embeddings

        # 2. Stack inputs: (R, s, a) -> (B, 3*T, n_embd)
        # We interleave them: R1, s1, a1, R2, s2, a2...
        # But for Decision Transformer, we usually predict a_t from (R_t, s_t) and history.
        # So we stack (R, s) for the current step, and 'a' from previous.
        # For simplicity in this implementation, we will treat the sequence as:
        # [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        
        # Reshape to (B, T, 1, n_embd) to stack
        stacked_inputs = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * T, self.config.n_embd)

        x = self.dropout(stacked_inputs)

        # 3. Transformer
        # Note: We need to handle the longer sequence length (3*T) in attention mask
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x) # (B, 3*T, n_embd)

        # 4. Predict Action
        # We want to predict action a_t given (R_t, s_t). This corresponds to the embedding of s_t in the sequence.
        # Sequence: R0, s0, a0, R1, s1, a1...
        # Indices:  0,  1,  2,  3,  4,  5...
        # s_t is at index 1, 4, 7... (which is 1 + 3*t)
        
        # Extract embeddings corresponding to state s_t
        # We want x[:, 1::3, :]
        
        state_reprs = x[:, 1::3, :] # (B, T, n_embd)
        
        action_preds = self.predict_action(state_reprs) # (B, T, act_dim)

        return action_preds
