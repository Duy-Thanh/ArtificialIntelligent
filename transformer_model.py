import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerConfig:
    """Configuration class for transformer model"""
    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 512,
        n_layer: int = 4,
        n_head: int = 8,
        n_embd: int = 256,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,  # Added this parameter
        bias: bool = True,
        **kwargs  # Allow for future extensibility
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon  # Store the new parameter
        self.bias = bias
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # key, query, value projections
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
        )
        
    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape attention_mask to match attention shape
            attention_mask = attention_mask.view(B, 1, 1, T)
            attention_mask = attention_mask.expand(-1, self.n_head, -1, -1)
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attention = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        b, t = input_ids.size()
        
        # Create position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Get token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(pos)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings.unsqueeze(0)
        x = self.dropout(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    attention_mask
                )
            else:
                x = block(x, attention_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0) 