"""
Discrete action space battery implementation.

This module contains the Decimal-based battery implementation optimized for
discrete action spaces with quantization support.
"""

from typing import Union
from decimal import Decimal, ROUND_CEILING
from ..base.battery import RechargeableBatteryBase


class RechargeableBatteryDiscrete(RechargeableBatteryBase):
    """
    Decimal-based rechargeable battery implementation for discrete action spaces.
    
    This implementation uses Python Decimal for precise calculations and includes
    quantization logic to ensure discrete step sizes are maintained.
    """
    
    def __init__(self, capacity, max_charging_rate, efficiency=None, init_soc=None, step_size=None):
        """
        Initialize discrete battery with quantization step size.
        
        Args:
            capacity: Total capacity of the battery in kWh.
            max_charging_rate: Maximum charging/discharging rate in kW.
            efficiency: Charging/discharging efficiency.
            init_soc: Initial state of charge as fraction of capacity.
            step_size: Step size for discretization in kW (default: 0.05 kW).
        """
        # Initialize step_size first as it's needed for validation
        self.step_size = self._validate_and_convert_initial(
            step_size if step_size is not None else Decimal("0.05")
        )
        
        # Call parent constructor
        super().__init__(capacity, max_charging_rate, efficiency, init_soc)
    
    def _validate_and_convert_initial(self, value) -> Decimal:
        """Initial validation for step_size before parent init."""
        if isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float, str)):
            return Decimal(str(value))
        else:
            raise TypeError(f"Expected numeric type for discrete battery, got {type(value)}")
    
    def _validate_and_convert(self, value) -> Decimal:
        """
        Validate and convert input to Decimal type.
        
        Args:
            value: Input value to convert.
            
        Returns:
            Value converted to Decimal.
            
        Raises:
            TypeError: If value cannot be converted to Decimal.
        """
        if isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float, str)):
            return Decimal(str(value))
        else:
            raise TypeError(f"Expected numeric type for discrete battery, got {type(value)}")
    
    def _get_default_efficiency(self) -> Decimal:
        """Get default efficiency for discrete battery."""
        return Decimal("1")
    
    def _get_default_init_soc(self) -> Decimal:
        """Get default initial state of charge for discrete battery."""
        return Decimal("0.5")
    
    def _convert_power_to_energy(self, power: Decimal, duration: Decimal) -> Decimal:
        """Convert power (kW) and duration (seconds) to energy (kWh)."""
        return power * (duration / Decimal("3600"))
    
    def _charge_impl(self, energy: Decimal, duration: Decimal) -> Decimal:
        """
        Charge implementation for discrete action space with quantization.
        
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
            return Decimal("0") / (duration / Decimal("3600"))
        
        if self.battery_soc + energy_to_be_charged > self.capacity:
            # Only charge up to the capacity
            energy_to_be_charged = self.capacity - self.battery_soc
            energy_consumed = energy_to_be_charged / self.efficiency  # energy_consumed is the lower limit drawn from the grid
            
            power_consumed = energy_consumed / (duration / Decimal("3600"))  # power to be charged in kW
            # Perform quantization on power_consumed, such that it is a multiple of step_size
            # and >= _power_consumed (if considering efficiency)
            # This is equivalent to a slightly decreased efficiency.
            power_consumed = (power_consumed / self.step_size).to_integral_exact(rounding=ROUND_CEILING) * self.step_size
            
            # Perform quantization on energy_to_be_charged
            energy_consumed = power_consumed * (duration / Decimal("3600"))
            
            # Update the battery state of charge
            self.battery_soc += energy_to_be_charged
        else:
            self.battery_soc += energy_to_be_charged
        
        return energy_consumed / (duration / Decimal("3600"))  # Return power in kW
    
    def _discharge_impl(self, energy: Decimal, duration: Decimal, y_t: Decimal) -> Decimal:
        """
        Discharge implementation for discrete action space with quantization.
        
        Args:
            energy: Energy in kWh to discharge from battery.
            duration: Duration in seconds for discharging.
            y_t: User load at current step.
            
        Returns:
            Power discharged in kW.
        """
        requested_energy_draw = -y_t * (duration / Decimal("3600")) / Decimal("1000")  # kWh
        
        energy_draw_from_battery = energy  # energy to be deduced from battery soc
        # Perform quantization on energy_discharged based on power quantization,
        # such that the output power is a multiple of step_size
        # and <= energy_discharged (when considering efficiency)
        energy_discharged = int(energy * self.efficiency / (duration / Decimal("3600")) / self.step_size) * self.step_size * (duration / Decimal("3600"))
        
        if abs(energy_discharged) > abs(requested_energy_draw):
            energy_discharged = requested_energy_draw
            energy_draw_from_battery = requested_energy_draw / self.efficiency
        
        new_battery_soc = self.battery_soc + energy_draw_from_battery
        
        if new_battery_soc < 0:
            energy_discharged = -(self.battery_soc * self.efficiency)
            power_discharged = energy_discharged / (duration / Decimal("3600"))  # power discharged in kW
            # Quantize the power discharged to the nearest step size
            # Note that for efficiency < 1, this calculation will result in a power_discharged
            # that is less than the power_discharged (above) due to the quantization.
            # This is equivalent to a slightly decreased efficiency.
            power_discharged = int(power_discharged / self.step_size) * self.step_size  # use integer to drop any decimal places
            energy_discharged = power_discharged * (duration / Decimal("3600"))  # recalculate energy discharged based on the quantized power
        
        self.battery_soc = max(new_battery_soc, Decimal("0"))
        return energy_discharged / (duration / Decimal("3600"))  # Return power in kW
    
    def compute_unnormalized_charge(self, normalized_action: Decimal) -> Decimal:
        """
        Compute unnormalized charge for discrete action space.
        
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
    
    def reset(self, init_soc=None):
        """
        Reset the battery to specified initial state of charge.
        
        Args:
            init_soc: Initial state of charge as fraction of capacity.
        """
        if init_soc is None:
            init_soc = Decimal("0.5")
        init_soc = self._validate_and_convert(init_soc)
        self.battery_soc = init_soc * self.capacity
    
    def get_battery_config(self):
        """
        Get battery configuration with step_size included.
        
        Returns:
            Dictionary containing battery configuration with float values for serialization.
        """
        config = super().get_battery_config()
        config['step_size'] = float(self.step_size)
        return config
