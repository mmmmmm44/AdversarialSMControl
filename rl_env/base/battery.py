"""
Abstract base class for rechargeable battery implementations.

This module defines the common interface and shared functionality for different
types of rechargeable batteries used in the Smart Meter RL environment.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Union, Dict, Any
from decimal import Decimal

# Type variable for numeric types used in battery calculations
NumericType = TypeVar('NumericType', float, Decimal)


class RechargeableBatteryBase(ABC):
    """
    Abstract base class for rechargeable battery systems.
    
    This class defines the common interface and shared functionality for battery
    implementations with different numeric precision requirements (float vs Decimal).
    """
    
    def __init__(self, capacity, max_charging_rate, efficiency=None, init_soc=None):
        """
        Initialize the rechargeable battery with given parameters.
        
        Args:
            capacity: Total capacity of the battery in kWh.
            max_charging_rate: Maximum charging/discharging rate in kW.
            efficiency: Charging/discharging efficiency (default varies by implementation).
            init_soc: Initial state of charge as a fraction of capacity (default varies by implementation).
        """
        # Type validation and conversion happens in concrete classes
        self.capacity = self._validate_and_convert(capacity)
        self.max_charging_rate = self._validate_and_convert(max_charging_rate)
        self.efficiency = self._validate_and_convert(
            efficiency if efficiency is not None else self._get_default_efficiency()
        )
        
        init_soc_val = self._validate_and_convert(
            init_soc if init_soc is not None else self._get_default_init_soc()
        )
        self.battery_soc = init_soc_val * self.capacity
    
    @abstractmethod
    def _validate_and_convert(self, value) -> NumericType:
        """
        Validate and convert input value to the appropriate numeric type.
        
        Args:
            value: Input value to validate and convert.
            
        Returns:
            Converted value in the appropriate numeric type (float or Decimal).
            
        Raises:
            TypeError: If the input value cannot be converted to the expected type.
        """
        pass
    
    @abstractmethod
    def _get_default_efficiency(self) -> NumericType:
        """Get the default efficiency value for this battery type."""
        pass
    
    @abstractmethod
    def _get_default_init_soc(self) -> NumericType:
        """Get the default initial state of charge for this battery type."""
        pass
    
    @abstractmethod
    def _convert_power_to_energy(self, power: NumericType, duration: NumericType) -> NumericType:
        """Convert power and duration to energy."""
        pass
    
    def charge_discharge(self, power, duration, y_t):
        """
        Template method for charging or discharging the battery.
        
        Args:
            power: Power in kW to charge or discharge the battery.
            duration: Duration in seconds for which the battery is charged/discharged.
            y_t: The user load (kW) at the current step.
            
        Returns:
            The power charged/discharged in kW, with efficiency considered.
        """
        # Validate and convert inputs using the concrete implementation's type system
        power = self._validate_and_convert(power)
        duration = self._validate_and_convert(duration)
        y_t = self._validate_and_convert(y_t)
        
        # Convert power to energy
        energy = self._convert_power_to_energy(power, duration)
        
        # Delegate to appropriate implementation based on power sign
        if power >= 0:  # Charging
            return self._charge_impl(energy, duration)
        else:  # Discharging
            return self._discharge_impl(energy, duration, y_t)
    
    @abstractmethod
    def _charge_impl(self, energy: NumericType, duration: NumericType) -> NumericType:
        """
        Implementation-specific charging logic.
        
        Args:
            energy: Energy in kWh to charge the battery.
            duration: Duration in seconds for charging.
            
        Returns:
            Power consumed from grid in kW.
        """
        pass
    
    @abstractmethod
    def _discharge_impl(self, energy: NumericType, duration: NumericType, y_t: NumericType) -> NumericType:
        """
        Implementation-specific discharging logic.
        
        Args:
            energy: Energy in kWh to discharge from the battery.
            duration: Duration in seconds for discharging.
            y_t: User load at current step.
            
        Returns:
            Power discharged in kW.
        """
        pass
    
    def get_normalized_state_of_charge(self) -> float:
        """
        Get the normalized state of charge (SoC) of the battery.
        
        Returns:
            Current state of charge as a fraction of capacity [0, 1].
        """
        # Always return float for compatibility with RL frameworks
        return float(self.battery_soc / self.capacity)
    
    @abstractmethod
    def compute_unnormalized_charge(self, normalized_action) -> NumericType:
        """
        Compute the unnormalized charge based on normalized action.
        
        Args:
            normalized_action: Normalized action value in range [-1, 1].
            
        Returns:
            Unnormalized charge in kW.
        """
        pass
    
    @abstractmethod
    def reset(self, init_soc=None):
        """
        Reset the battery to a specified initial state of charge.
        
        Args:
            init_soc: Initial state of charge as a fraction of capacity.
        """
        pass
    
    def get_battery_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the battery as a dictionary.
        
        Returns:
            Dictionary containing battery configuration with float values for serialization.
        """
        return {
            'capacity': float(self.capacity),
            'max_charging_rate': float(self.max_charging_rate),
            'efficiency': float(self.efficiency),
            'initial_soc': float(self.battery_soc / self.capacity)
        }


class BatteryFactory:
    """Factory class for creating battery instances."""
    
    @staticmethod
    def create(action_type: str, **kwargs) -> RechargeableBatteryBase:
        """
        Create a battery instance based on action type.
        
        Args:
            action_type: Type of action space ('continuous' or 'discrete').
            **kwargs: Battery configuration parameters.
            
        Returns:
            Battery instance appropriate for the action type.
            
        Raises:
            ValueError: If action_type is not supported.
        """
        if action_type.lower() == 'continuous':
            from ..continuous.battery import RechargeableBattery
            return RechargeableBattery(**kwargs)
        elif action_type.lower() == 'discrete':
            from ..discrete.battery import RechargeableBatteryDiscrete
            return RechargeableBatteryDiscrete(**kwargs)
        else:
            raise ValueError(f"Unsupported action type: {action_type}. Must be 'continuous' or 'discrete'.")
