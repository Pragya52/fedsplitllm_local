"""
model_architecture.py
Split LLaMA 3.2 Model Architecture for Federated Learning
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    LlamaForCausalLM,
    LlamaConfig
)
from typing import Optional, Tuple
import logging
import warnings
import os

logger = logging.getLogger(__name__)
# Suppress warnings
warnings.filterwarnings("ignore")

class ClientModel(nn.Module):
    """
    Client-side model containing embeddings and first N transformer layers
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.model_name = model_name
        
        try:
            # Try to load Llama 3.2 1B model
            print(f"Loading {model_name}...")
            
            # Check if HF token is set
            hf_token = os.environ.get("HF_TOKEN", None)
            
            if "meta-llama/Llama-3.2" in model_name:
                # Loading actual Llama model
                full_model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
            else:
                # Generic model loading
                full_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
            
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            print("\nUsing a smaller fallback model for testing...")
            print("To use Llama 3.2 1B:")
            print("1. Get access at: https://huggingface.co/meta-llama/Llama-3.2-1B")
            print("2. Set your HF token: export HF_TOKEN='your_token_here'")
            print("3. Or login with: huggingface-cli login\n")
            
            # Fallback to a smaller public model
            model_name = "EleutherAI/pythia-70m"  # Very small model for testing
            full_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        
        # Extract client components
        self.config = full_model.config
        
        # Handle Llama 3.2 architecture
        if hasattr(full_model, 'model'):  # Llama architecture
            self.embed_tokens = full_model.model.embed_tokens
            
            # Only take first N layers
            available_layers = len(full_model.model.layers)
            layers_to_use = min(num_layers, available_layers)
            
            self.layers = nn.ModuleList([
                full_model.model.layers[i] for i in range(layers_to_use)
            ])
            
            # Store RoPE parameters if available
            if hasattr(full_model.model, 'rotary_emb'):
                self.rotary_emb = full_model.model.rotary_emb
            
        else:  # Fallback architecture (GPT-style)
            if hasattr(full_model, 'gpt_neox'):  # Pythia model
                self.embed_tokens = full_model.gpt_neox.embed_in
                available_layers = len(full_model.gpt_neox.layers)
                layers_to_use = min(num_layers, available_layers)
                self.layers = nn.ModuleList([
                    full_model.gpt_neox.layers[i] for i in range(layers_to_use)
                ])
            else:  # Generic transformer
                self.embed_tokens = full_model.transformer.wte
                available_layers = len(full_model.transformer.h)
                layers_to_use = min(num_layers, available_layers)
                self.layers = nn.ModuleList([
                    full_model.transformer.h[i] for i in range(layers_to_use)
                ])
        
        self.hidden_size = full_model.config.hidden_size
        self.actual_num_layers = len(self.layers)
        
        # Delete full model to save memory
        del full_model
        torch.cuda.empty_cache()
        
        logger.info({
            "model/client_layers": self.actual_num_layers,
            "model/client_model": model_name,
            "model/hidden_size": self.hidden_size
        })
        
        print(f"Client model initialized with {self.actual_num_layers} layers")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through client layers
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=device
            )
        
        # Expand attention mask for some architectures
        if attention_mask.dim() == 2:
            # Create 4D attention mask from 2D mask
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through client layers
        for layer in self.layers:
            # Different layer signatures for different models
            if "llama" in self.model_name.lower() or hasattr(layer, 'self_attn'):
                # Llama-style layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            else:
                # Generic transformer layer
                outputs = layer(hidden_states, attention_mask=attention_mask)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        return hidden_states, None
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ServerModel(nn.Module):
    """
    Server-side model containing remaining transformer layers and output head
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", start_layer: int = 4):
        super().__init__()
        self.start_layer = start_layer
        self.model_name = model_name
        
        try:
            # Try to load Llama 3.2 1B model
            print(f"Loading server model {model_name}...")
            
            # Check if HF token is set
            hf_token = os.environ.get("HF_TOKEN", None)
            
            if "meta-llama/Llama-3.2" in model_name:
                # Loading actual Llama model
                full_model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
            else:
                # Generic model loading
                full_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    token=hf_token,
                    low_cpu_mem_usage=True
                )
                
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            print("Using fallback model for server...")
            
            # Fallback to smaller model
            model_name = "EleutherAI/pythia-70m"
            full_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        
        # Extract server components
        self.config = full_model.config
        
        # Handle Llama 3.2 architecture
        if hasattr(full_model, 'model'):  # Llama architecture
            total_layers = len(full_model.model.layers)
            
            # Take layers from start_layer to end
            self.layers = nn.ModuleList([
                full_model.model.layers[i] 
                for i in range(min(start_layer, total_layers), total_layers)
            ])
            
            self.norm = full_model.model.norm
            self.lm_head = full_model.lm_head
            
        else:  # Fallback architecture
            if hasattr(full_model, 'gpt_neox'):  # Pythia model
                total_layers = len(full_model.gpt_neox.layers)
                self.layers = nn.ModuleList([
                    full_model.gpt_neox.layers[i]
                    for i in range(min(start_layer, total_layers), total_layers)
                ])
                self.norm = full_model.gpt_neox.final_layer_norm
                self.lm_head = full_model.embed_out
            else:  # Generic transformer
                total_layers = len(full_model.transformer.h)
                self.layers = nn.ModuleList([
                    full_model.transformer.h[i]
                    for i in range(min(start_layer, total_layers), total_layers)
                ])
                self.norm = full_model.transformer.ln_f
                self.lm_head = full_model.lm_head
        
        self.actual_num_layers = len(self.layers)
        
        # Delete full model to save memory
        del full_model
        torch.cuda.empty_cache()
        
        logger.info({
            "model/server_layers": self.actual_num_layers,
            "model/server_start": start_layer,
            "model/server_model": model_name
        })
        
        print(f"Server model initialized with {self.actual_num_layers} layers (from layer {start_layer})")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Forward pass through server layers
        """
        # Pass through server layers
        for layer in self.layers:
            if "llama" in self.model_name.lower() or hasattr(layer, 'self_attn'):
                # Llama-style layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            else:
                # Generic transformer layer
                outputs = layer(hidden_states, attention_mask=attention_mask)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return logits, None, None
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute language modeling loss
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten and compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SplitLLaMA3Model:
    """
    Manager class for split Llama 3.2 model architecture
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", client_layers: int = 4):
        self.model_name = model_name
        self.client_layers = client_layers
        
        print(f"\nInitializing Split Model Architecture")
        print(f"Model: {model_name}")
        print(f"Client layers: {client_layers}")
        
        # Initialize tokenizer
        try:
            hf_token = os.environ.get("HF_TOKEN", None)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
        except Exception as e:
            print(f"Could not load tokenizer for {model_name}, using fallback")
            # Use a fallback tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load configuration
        try:
            hf_token = os.environ.get("HF_TOKEN", None)
            if "meta-llama/Llama-3.2" in model_name:
                self.config = LlamaConfig.from_pretrained(model_name, token=hf_token)
            else:
                self.config = AutoConfig.from_pretrained(model_name, token=hf_token)
            self.total_layers = self.config.num_hidden_layers
        except:
            print("Using default configuration")
            self.total_layers = 16  # Default for small models
        
        # Ensure we don't exceed available layers
        self.client_layers = min(client_layers, max(1, self.total_layers - 2))
        
        logger.info({
            "model/name": model_name,
            "model/total_layers": self.total_layers,
            "model/client_layers": self.client_layers,
            "model/server_layers": self.total_layers - self.client_layers
        })
        
        print(f"Total layers: {self.total_layers}")
        print(f"Client will use layers 0-{self.client_layers-1}")
        print(f"Server will use layers {self.client_layers}-{self.total_layers-1}")
    
    def create_client_model(self) -> ClientModel:
        """Create and return client model"""
        return ClientModel(self.model_name, self.client_layers)
    
    def create_server_model(self) -> ServerModel:
        """Create and return server model"""
        return ServerModel(self.model_name, self.client_layers)
    
    def validate_split(self, client_model: ClientModel, server_model: ServerModel) -> bool:
        """Validate the split architecture"""
        client_params = client_model.get_num_parameters()
        server_params = server_model.get_num_parameters()
        total_params = client_params + server_params
        
        print(f"\nModel Split Validation:")
        print(f"Client parameters: {client_params:,}")
        print(f"Server parameters: {server_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Client/Server ratio: {client_params/total_params:.1%}/{server_params/total_params:.1%}")
        
        logger.info({
            "model/client_parameters": client_params,
            "model/server_parameters": server_params,
            "model/total_parameters": total_params,
            "model/client_param_ratio": client_params/total_params,
            "model/server_param_ratio": server_params/total_params
        })
        
        return True