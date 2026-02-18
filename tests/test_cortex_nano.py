import torch
import pytest
from celr.cortex.model_nano import NanoCortex, NanoConfig

def test_nano_cortex_forward_pass():
    """
    Verifies that NanoCortex accepts inputs of correct shape
    and produces logits of correct shape.
    """
    state_dim = 8
    act_dim = 7
    max_len = 10
    batch_size = 2
    
    config = NanoConfig(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=max_len,
        n_layer=2,
        n_head=2,
        n_embd=32
    )
    model = NanoCortex(config)
    
    # Create dummy inputs
    # States: (B, T, 8)
    states = torch.randn(batch_size, max_len, state_dim)
    # Actions: (B, T)
    actions = torch.randint(0, act_dim, (batch_size, max_len))
    # Returns: (B, T, 1)
    returns = torch.randn(batch_size, max_len, 1)
    # Timesteps: (B, T)
    timesteps = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1)
    
    # Forward
    logits = model(states, actions, returns, timesteps)
    
    # Check output
    assert logits.shape == (batch_size, max_len, act_dim)
    print("NanoCortex Forward Pass: OK")

def test_overfitting_single_batch():
    """
    Verifies that the model can actually learn (loss decreases) on a dummy batch.
    """
    state_dim = 8
    act_dim = 7
    max_len = 5
    
    config = NanoConfig(state_dim=state_dim, act_dim=act_dim, max_length=max_len, n_embd=32)
    model = NanoCortex(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Dummy data
    states = torch.randn(1, max_len, state_dim)
    actions = torch.randint(0, act_dim, (1, max_len))
    returns = torch.randn(1, max_len, 1)
    timesteps = torch.arange(max_len).unsqueeze(0)
    
    # Train for 10 steps
    initial_loss = None
    final_loss = None
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for i in range(20):
        logits = model(states, actions, returns, timesteps)
        # Reshape for loss
        # logits (B, T, C) -> (B*T, C)
        # actions (B, T) -> (B*T)
        loss = loss_fn(logits.view(-1, act_dim), actions.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        
    print(f"Initial Loss: {initial_loss}, Final Loss: {final_loss}")
    assert final_loss < initial_loss
