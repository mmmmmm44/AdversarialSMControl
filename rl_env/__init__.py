"""
Smart Meter RL Environment Package.

This package provides reinforcement learning environments for smart meter
energy management with rechargeable battery systems. It supports both
continuous and discrete action spaces.

Usage:
    # Continuous action space
    from rl_env.continuous import RechargeableBattery
    from rl_env.common import create_battery
    
    # Discrete action space  
    from rl_env.discrete import RechargeableBatteryDiscrete
    
    # Factory pattern
    battery = create_battery('continuous', capacity=8.0, max_charging_rate=4.0)
"""

# Import key components for easy access
from .base import BatteryFactory
from .common import create_battery
from .continuous import RechargeableBattery
from .discrete import RechargeableBatteryDiscrete

__all__ = [
    'BatteryFactory',
    'create_battery', 
    'RechargeableBattery',
    'RechargeableBatteryDiscrete',
]

__version__ = '0.1.0'
