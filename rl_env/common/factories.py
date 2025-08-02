"""
Unified Factory Functions for RL Environment Components.

This module provides convenient factory functions to create different types of components
based on action space type. These functions serve as convenient wrappers around the
domain-specific factory classes, providing a functional interface and additional
features like default value management.

The core creation logic is delegated to domain-specific factories (e.g., BatteryFactory,
EpisodeFactory) to maintain a single source of truth while providing multiple convenient
interfaces. All factory functions follow consistent patterns for error handling, type
safety, and default value management.

Organization:
- Battery Factories: create_battery, create_battery_with_defaults
- Episode Factories: create_episode, create_episode_with_defaults  
- H-Network Factories: (Phase 3)
- Environment Factories: (Phase 4)
- Callback Factories: (Phase 5)
- Integrated Factories: (Future)
"""

from typing import Dict, Any
from decimal import Decimal
import pandas as pd


# ================================
# SHARED HELPER FUNCTIONS
# ================================

def _validate_action_type(action_type: str) -> str:
    """
    Validate and normalize action type.
    
    Args:
        action_type: The action type to validate
        
    Returns:
        Normalized action type
        
    Raises:
        ValueError: If action type is not supported
    """
    normalized = action_type.lower().strip()
    if normalized not in ('continuous', 'discrete'):
        raise ValueError(f"Unsupported action type: {action_type}. Must be 'continuous' or 'discrete'.")
    return normalized



# ================================
# BATTERY FACTORIES
# ================================

def create_battery(action_type: str, **kwargs):
    """
    Create a battery instance based on action type.
    
    This is a convenience wrapper around BatteryFactory.create() that provides
    a functional interface for battery creation.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        **kwargs: Battery configuration parameters.
        
    Returns:
        Battery instance appropriate for the action type.
        
    Raises:
        ValueError: If action_type is not supported.
        
    Example:
        >>> battery = create_battery('continuous', capacity=8.0, max_charging_rate=4.0)
        >>> battery = create_battery('discrete', 
        ...                        capacity=Decimal('8.0'), 
        ...                        max_charging_rate=Decimal('4.0'),
        ...                        step_size=Decimal('0.05'))
    """
    action_type = _validate_action_type(action_type)
    from ..base.battery import BatteryFactory
    return BatteryFactory.create(action_type, **kwargs)


def _get_battery_defaults(action_type: str) -> Dict[str, Any]:
    """
    Get default battery configuration for the given action type.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        
    Returns:
        Dictionary of default configuration values.
        
    Raises:
        ValueError: If action_type is not supported.
    """
    action_type = _validate_action_type(action_type)
    if action_type == 'continuous':
        return {
            'capacity': 8.0,
            'max_charging_rate': 4.0,
            'efficiency': 1.0,
            'init_soc': 0.5
        }
    else:  # discrete
        return {
            'capacity': Decimal('8.0'),
            'max_charging_rate': Decimal('4.0'),
            'efficiency': Decimal('1.0'),
            'init_soc': Decimal('0.5'),
            'step_size': Decimal('0.05')
        }


def _process_overrides_for_discrete(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process override values for discrete action type, converting to Decimal where appropriate.
    
    Args:
        overrides: Dictionary of override values.
        
    Returns:
        Dictionary with values converted to appropriate types for discrete batteries.
    """
    processed = {}
    for key, value in overrides.items():
        # Convert numeric values to Decimal for discrete batteries, except init_soc
        if isinstance(value, (int, float, str)) and key != 'init_soc':
            try:
                processed[key] = Decimal(str(value))
            except (ValueError, TypeError):
                processed[key] = value
        else:
            processed[key] = value
    return processed


def create_battery_with_defaults(action_type: str, **overrides) -> Any:
    """
    Create a battery with sensible defaults for the given action type.
    
    This function provides default configuration values for each action type
    and allows selective overriding of specific parameters. All creation
    logic is delegated to BatteryFactory.create() via create_battery().
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        **overrides: Configuration parameters to override defaults.
        
    Returns:
        Battery instance with default configuration and applied overrides.
        
    Example:
        >>> # Create continuous battery with defaults
        >>> battery = create_battery_with_defaults('continuous')
        >>> 
        >>> # Create discrete battery with custom capacity
        >>> battery = create_battery_with_defaults('discrete', capacity=10.0)  # Auto-converted to Decimal
    """
    action_type = _validate_action_type(action_type)
    
    # Get defaults for the action type
    defaults = _get_battery_defaults(action_type)
    
    # Process overrides (convert to appropriate types for discrete)
    if action_type == 'discrete':
        processed_overrides = _process_overrides_for_discrete(overrides)
    else:
        processed_overrides = overrides
    
    # Apply overrides to defaults
    defaults.update(processed_overrides)
    
    # Delegate to create_battery, which delegates to BatteryFactory.create()
    return create_battery(action_type, **defaults)


# ================================
# EPISODE FACTORIES
# ================================


def create_episode(action_type: str, selected_aggregate_load_df, **kwargs):
    """
    Create an episode instance based on action type.
    
    This is a convenience wrapper around EpisodeFactory.create() that provides
    a functional interface for episode creation.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        selected_aggregate_load_df: The DataFrame containing aggregate load data.
        **kwargs: Episode configuration parameters (e.g., step_size for discrete).
        
    Returns:
        Episode instance appropriate for the action type.
        
    Raises:
        ValueError: If action_type is not supported.
        
    Example:
        >>> episode = create_episode('continuous', load_df)
        >>> episode = create_episode('discrete', load_df, step_size=50)
    """
    action_type = _validate_action_type(action_type)
    from ..base.episode import EpisodeFactory
    return EpisodeFactory.create(action_type, selected_aggregate_load_df, **kwargs)


def _get_episode_defaults(action_type: str) -> Dict[str, Any]:
    """
    Get default episode configuration for the given action type.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        
    Returns:
        Dictionary of default configuration values.
    """
    action_type = _validate_action_type(action_type)
    if action_type == 'discrete':
        return {'step_size': 50}
    else:  # continuous
        return {}


def create_episode_with_defaults(action_type: str, selected_aggregate_load_df, **overrides):
    """
    Create an episode with sensible defaults for the given action type.
    
    This function provides default configuration values for episode parameters
    and allows selective overriding of specific parameters.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        selected_aggregate_load_df: The DataFrame containing aggregate load data.
        **overrides: Configuration parameters to override defaults.
            For discrete: step_size (int, default=50)
        
    Returns:
        Episode instance with default configuration and applied overrides.
        
    Example:model/H_network
        >>> # Create continuous episode (no special defaults)
        >>> episode = create_episode_with_defaults('continuous', load_df)
        >>> 
        >>> # Create discrete episode with custom step size
        >>> episode = create_episode_with_defaults('discrete', load_df, step_size=100)
    """
    action_type = _validate_action_type(action_type)
    
    # Get defaults for episode parameters
    defaults = _get_episode_defaults(action_type)
    
    # Merge defaults with overrides
    params = {**defaults, **overrides}
    
    # Delegate to create_episode, which delegates to EpisodeFactory.create()
    return create_episode(action_type, selected_aggregate_load_df, **params)


# ================================
# H-NETWORK FACTORIES
# ================================

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
    from model.H_network.common.factories import create_h_network_module as _create_h_network_module
    return _create_h_network_module(action_type, h_network_type, device, 
                                  optimizer_class, optimizer_kwargs, **kwargs)


def create_h_network_module_with_defaults(action_type: str, **kwargs):
    """
    Create an H-Network module with default settings.
    
    Args:
        action_type: 'continuous' or 'discrete'
        **kwargs: Additional arguments to override defaults
        
    Returns:
        H-Network module instance with default configuration
    """
    from model.H_network.common.factories import create_h_network_module_with_defaults as _create_h_network_module_with_defaults
    return _create_h_network_module_with_defaults(action_type, **kwargs)
def create_h_network_module_with_defaults(action_type: str, **overrides):
    """
    Create an H-Network RL module with sensible defaults for the given action type.
    
    This function provides default configuration values for H-Network parameters
    and allows selective overriding of specific parameters.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        **overrides: Configuration parameters to override defaults.
            Common: h_network_type, device
            For discrete: output_dim (int, default=101)
        
    Returns:
        H-Network RL module instance with default configuration and applied overrides.
        
    Example:
        >>> # Create continuous H-Network with defaults
        >>> h_module = create_h_network_module_with_defaults('continuous')
        >>> 
        >>> # Create discrete H-Network with custom device
        >>> h_module = create_h_network_module_with_defaults('discrete', device='cuda')
    """
    action_type = _validate_action_type(action_type)
    from model.H_network.common.factories import create_h_network_module_with_defaults as _create_with_defaults
    return _create_with_defaults(action_type, **overrides)


# ================================
# ENVIRONMENT FACTORIES  
# ================================

def create_environment(action_type: str, **kwargs):
    """
    Create a Smart Meter environment for the given action type.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        **kwargs: Configuration parameters passed to environment constructor.
            Required parameters:
            - smart_meter_data_loader: Data loader for smart meter data
            - h_network_rl_module: H-network module for privacy computation
            - mode: Training mode (TrainingMode enum)
            - log_folder: Path for logging
            Optional parameters:
            - rb_config (dict): Battery configuration parameters
            - reward_lambda (float): Weighting factor for reward function (default: 0.5)
            - render_mode: Rendering mode ('human' or None)
            - render_host (str): Host address for render server (default: '127.0.0.1')
            - render_port (int): Port for render server (default: 50007)
        
    Returns:
        Smart Meter environment instance for the specified action type.
        
    Raises:
        ValueError: If action type is not supported.
        
    Example:
        >>> from rl_env.data_loader import SimpleSmartMeterDataLoader
        >>> from rl_env.training_mode import TrainingMode
        >>> from pathlib import Path
        >>> 
        >>> # Create continuous environment
        >>> env = create_environment('continuous', 
        ...                        smart_meter_data_loader=data_loader,
        ...                        h_network_rl_module=h_network_module,
        ...                        mode=TrainingMode.TRAIN,
        ...                        log_folder=Path('logs'))
        >>> 
        >>> # Create discrete environment with custom config
        >>> env = create_environment('discrete', 
        ...                        smart_meter_data_loader=data_loader,
        ...                        h_network_rl_module=h_network_module,
        ...                        mode=TrainingMode.TRAIN,
        ...                        log_folder=Path('logs'),
        ...                        render_mode='human',
        ...                        rb_config={'capacity': 10.0})
    """
    action_type = _validate_action_type(action_type)
    
    if action_type == 'continuous':
        from ..continuous.env_module import SmartMeterContinuousEnv
        return SmartMeterContinuousEnv(**kwargs)
    elif action_type == 'discrete':
        from ..discrete.env_module import SmartMeterDiscreteEnv
        return SmartMeterDiscreteEnv(**kwargs)
    else:
        raise ValueError(f"Unsupported action type: {action_type}")
    
def _get_environment_defaults(action_type: str) -> Dict[str, Any]:
    """
    Get default environment configuration for the given action type.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        
    Returns:
        Dictionary of default configuration values.
    """
    action_type = _validate_action_type(action_type)

    defaults = {
        'rb_config': None,
        'reward_lambda': 0.5,
        'render_mode': None,
        'render_host': '127.0.0.1',
        'render_port': 50007
    }
    
    if action_type == 'discrete':
        defaults.update({
            'aggregate_step_size': 50,
            'battery_step_size': Decimal("0.05")
        })

    return defaults



def create_environment_with_defaults(action_type: str, smart_meter_data_loader, h_network_rl_module, mode, log_folder, **overrides):
    """
    Create a Smart Meter environment with sensible defaults for the given action type.
    
    This function provides default configuration values for environment parameters
    and allows selective overriding of specific parameters.
    
    Args:
        action_type: Type of action space ('continuous' or 'discrete').
        smart_meter_data_loader: Required data loader for smart meter data.
        h_network_rl_module: Required H-network module for privacy computation.
        mode: Required training mode (TrainingMode enum).
        log_folder: Required path for logging.
        **overrides: Configuration parameters to override defaults.
            Parameters with defaults:
            - rb_config (dict): None (uses default battery config)
            - reward_lambda (float): 0.5
            - render_mode (str): None
            - render_host (str): '127.0.0.1'
            - render_port (int): 50007
        
    Returns:
        Smart Meter environment instance with default configuration and applied overrides.
        
    Example:
        >>> from rl_env.data_loader import SimpleSmartMeterDataLoader
        >>> from rl_env.training_mode import TrainingMode
        >>> from pathlib import Path
        >>> 
        >>> # Create continuous environment with defaults
        >>> env = create_environment_with_defaults('continuous', 
        ...                                       data_loader, 
        ...                                       h_network_module,
        ...                                       TrainingMode.TRAIN,
        ...                                       Path('logs'))
        >>> 
        >>> # Create discrete environment with custom render mode
        >>> env = create_environment_with_defaults('discrete', 
        ...                                       data_loader,
        ...                                       h_network_module, 
        ...                                       TrainingMode.TRAIN,
        ...                                       Path('logs'),
        ...                                       render_mode='human',
        ...                                       reward_lambda=0.3)
    """
    action_type = _validate_action_type(action_type)
    defaults = _get_environment_defaults(action_type)
    
    # Apply overrides to defaults
    config = {**defaults, **overrides}
    
    # Add required parameters
    config.update({
        'smart_meter_data_loader': smart_meter_data_loader,
        'h_network_rl_module': h_network_rl_module,
        'mode': mode,
        'log_folder': log_folder
    })
    
    return create_environment(action_type, **config)

# ================================
# INTEGRATED FACTORIES
# ================================
# TODO: Future - Add integrated setup functions here
