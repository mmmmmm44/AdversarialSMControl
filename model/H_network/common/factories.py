"""
Factory functions for H-Network module creation.

This module provides convenient factory functions for creating H-Network instances
with proper type safety and defaults. These functions delegate to the core
HNetworkModuleFactory for consistency.
"""

from typing import Any, Dict
from pathlib import Path

from ..h_network_arch import HNetworkType


def create_h_network_module(action_type: str, h_network_type, device: str,
                           optimizer_class=None, optimizer_kwargs=None, **kwargs):
    """
    Create an H-Network module based on the action type.
    
    Args:
        action_type: 'continuous' or 'discrete'
        h_network_type: Type of H-Network architecture
        device: Device to run on ('cpu' or 'cuda')
        optimizer_class: Optimizer class (default: torch.optim.Adam)
        optimizer_kwargs: Dict of optimizer arguments (default: {'lr': 1e-3})
        **kwargs: Additional arguments (e.g., output_dim for discrete)
        
    Returns:
        H-Network module instance
    """
    if action_type == 'continuous':
        from ..continuous.h_network_module import HNetworkRLModule
        return HNetworkRLModule(h_network_type, device, optimizer_class, optimizer_kwargs, **kwargs)
    elif action_type == 'discrete':
        from ..discrete.h_network_module import HNetworkDiscreteRLModule
        return HNetworkDiscreteRLModule(h_network_type, device, optimizer_class, optimizer_kwargs, **kwargs)
    else:
        raise ValueError(f"Unknown action type: {action_type}. Must be 'continuous' or 'discrete'.")


def create_h_network_module_with_defaults(action_type: str, **kwargs):
    """
    Create an H-Network module with default settings.
    
    Args:
        action_type: 'continuous' or 'discrete'
        **kwargs: Additional arguments to override defaults
        
    Returns:
        H-Network module instance with default configuration
    """
    import torch
    
    defaults = {
        'h_network_type': HNetworkType.H_NETWORK,
        'device': 'cpu',
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 1e-3}
    }
    
    # Override defaults with provided kwargs
    config = {**defaults, **kwargs}
    
    return create_h_network_module(action_type, **config)

def _get_h_network_defaults(action_type: str) -> Dict[str, Any]:
    """
    Get default H-Network configuration for the given action type.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete')
        
    Returns:
        Dictionary of default configuration values
    """
    action_type = action_type.lower().strip()
    
    if action_type == 'continuous':
        return {
            'h_network_type': HNetworkType.H_NETWORK,
            'device': 'cpu'
        }
    elif action_type == 'discrete':
        return {
            'h_network_type': HNetworkType.H_NETWORK,
            'device': 'cpu',
            'output_dim': 101       # example default for discrete observation space
        }
    else:
        raise ValueError(f"Unsupported action type: {action_type}. Must be 'continuous' or 'discrete'.")


def create_h_network_module_with_defaults(action_type: str, **overrides) -> Any:
    """
    Create an H-Network RL module with sensible defaults for the given action type.
    
    This function provides default configuration values for H-Network parameters
    and allows selective overriding of specific parameters.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete')
        **overrides: Configuration parameters to override defaults
            Common: h_network_type, device
            For discrete: output_dim (int, default=101)
        
    Returns:
        H-Network RL module instance with default configuration and applied overrides
        
    Example:
        >>> # Create continuous H-Network with defaults
        >>> h_module = create_h_network_module_with_defaults('continuous')
        >>> 
        >>> # Create discrete H-Network with custom output dimension
        >>> h_module = create_h_network_module_with_defaults('discrete', output_dim=51, device='cuda')
    """
    # Get defaults for the action type
    defaults = _get_h_network_defaults(action_type)
    
    # Merge defaults with overrides
    params = {**defaults, **overrides}
    
    # Extract parameters for factory call
    h_network_type = params.pop('h_network_type')
    device = params.pop('device')
    
    # Delegate to create_h_network_module
    return create_h_network_module(action_type, h_network_type, device, **params)


def load_h_network_module_from_checkpoint(action_type: str, h_network_type: HNetworkType, 
                                        device: str, checkpoint_path: Path, **kwargs) -> Any:
    """
    Create and load an H-Network RL module from a saved checkpoint.
    
    Args:
        action_type: The type of action space ('continuous' or 'discrete')
        h_network_type: The type of H-network architecture to use
        device: Device to run the H-network on
        checkpoint_path: Path to the saved H-network checkpoint
        **kwargs: Additional arguments for module creation
        
    Returns:
        H-Network RL module instance loaded from checkpoint
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If loading fails
    """
    # Create the module
    h_module = create_h_network_module(action_type, h_network_type, device, **kwargs)
    
    # Initialize the network (required before loading)
    h_module.initialize_h_network(**kwargs)
    
    # Load the checkpoint
    h_module.load_h_network(checkpoint_path)
    
    return h_module
