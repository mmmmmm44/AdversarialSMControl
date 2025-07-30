"""
Continuous action space battery implementation.

This module contains the float-based battery implementation optimized for
continuous action spaces in reinforcement learning environments.
"""


from decimal import Decimal
from ..base.battery import RechargeableBatteryBase


class RechargeableBattery(RechargeableBatteryBase):
    """
    Float-based rechargeable battery implementation for continuous action spaces.
    
    This implementation uses standard Python floats for all calculations,
    providing good performance for continuous RL algorithms like PPO.
    """
    
    def _validate_and_convert(self, value) -> float:
        """
        Validate and convert input to float type.
        
        Args:
            value: Input value to convert.
            
        Returns:
            Value converted to float.
            
        Raises:
            TypeError: If value cannot be converted to float.
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise TypeError(f"Cannot convert string '{value}' to float")
        else:
            raise TypeError(f"Expected numeric type, got {type(value)}")
    
    def _get_default_efficiency(self) -> float:
        """Get default efficiency for continuous battery."""
        return 1.0
    
    def _get_default_init_soc(self) -> float:
        """Get default initial state of charge for continuous battery."""
        return 0.5
    
    def _convert_power_to_energy(self, power: float, duration: float) -> float:
        """Convert power (kW) and duration (seconds) to energy (kWh)."""
        return power * (duration / 3600)
    
    def _charge_impl(self, energy: float, duration: float) -> float:
        """
        Charge implementation for continuous action space.
        
        Args:
            energy: Energy in kWh to charge the battery.
            duration: Duration in seconds for charging.
            
        Returns:
            Power consumed from grid in kW.
        """
        energy_to_be_charged = energy * self.efficiency  # energy to be charged to battery soc
        energy_consumed = energy  # energy consumed from the grid
        
        if self.battery_soc >= self.capacity:
            # Battery is already full, no energy can be charged
            return 0.0 / (duration / 3600)
        
        if self.battery_soc + energy_to_be_charged > self.capacity:
            # Only charge up to capacity
            energy_to_be_charged = self.capacity - self.battery_soc
            energy_consumed = energy_to_be_charged / self.efficiency
            self.battery_soc = self.capacity
        else:
            self.battery_soc += energy_to_be_charged
        
        return energy_consumed / (duration / 3600)  # Return power in kW
    
    def _discharge_impl(self, energy: float, duration: float, y_t: float) -> float:
        """
        Discharge implementation for continuous action space.
        
        Args:
            energy: Energy in kWh to discharge from battery.
            duration: Duration in seconds for discharging.
            y_t: User load at current step.
            
        Returns:
            Power discharged in kW.
        """
        requested_energy_draw = -y_t * (duration / 3600) / 1000  # kWh
        
        energy_draw_from_battery = energy  # energy to be deduced from battery soc
        energy_discharged = energy * self.efficiency  # energy discharged to support user load
        
        if abs(energy_discharged) > abs(requested_energy_draw):
            energy_discharged = requested_energy_draw
            energy_draw_from_battery = requested_energy_draw / self.efficiency
        
        new_battery_soc = self.battery_soc + energy_draw_from_battery
        
        if new_battery_soc < 0:
            energy_discharged = -(self.battery_soc * self.efficiency)
        
        self.battery_soc = max(new_battery_soc, 0.0)
        return energy_discharged / (duration / 3600)  # Return power in kW
    
    def compute_unnormalized_charge(self, normalized_action: float) -> float:
        """
        Compute unnormalized charge for continuous action space.
        
        Args:
            normalized_action: Normalized action value in range [-1, 1].
            
        Returns:
            Unnormalized charge in kW.
        """
        normalized_action = self._validate_and_convert(normalized_action)
        
        if normalized_action < 0:
            # Discharging
            return normalized_action * self.max_charging_rate
        else:
            # Charging
            return normalized_action * self.max_charging_rate
    
    def reset(self, init_soc: float = 0.5):
        """
        Reset the battery to specified initial state of charge.
        
        Args:
            init_soc: Initial state of charge as fraction of capacity.
        """
        init_soc = self._validate_and_convert(init_soc) if init_soc is not None else 0.5
        self.battery_soc = init_soc * self.capacity
