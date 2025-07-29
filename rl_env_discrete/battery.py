"""
The rechargeable battery module simulates a rechargeable battery system.
It provides methods to charge, discharge, and check the battery's state of charge (SoC).
It also includes methods to calculate the energy stored in the battery and to reset its state.
"""

from decimal import Decimal, ROUND_CEILING


class RechargeableBatteryDiscrete:

    def __init__(self, capacity:Decimal, max_charging_rate:Decimal, efficiency=Decimal("1"), init_soc=Decimal("0.5"), step_size=Decimal("0.05")):
        """
        Initializes the rechargeable battery with given parameters.
        Parameters:
            capacity (decimal): Total capacity of the battery in Wh.
            max_charging_rate (decimal): Maximum charging/discharging rate in kW.
            efficiency (decimal): Efficiency of charging/discharging (default is 1, i.e., 100%).
            init_soc (decimal): Initial state of charge (SoC) as a fraction of capacity (default is 0.5).
            step_size (decimal): Step size for discretization (default is 0.05 kW).
        """

        assert type(capacity) is Decimal, "Capacity must be a Decimal."
        assert type(max_charging_rate) is Decimal, "Max charging rate must be a Decimal."
        assert type(efficiency) is Decimal, "Efficiency must be a Decimal."
        assert type(init_soc) is Decimal, "Initial state of charge must be a Decimal."
        assert type(step_size) is Decimal, "Step size must be a Decimal."

        self.capacity = capacity  # Total capacity in kWh
        self.max_charging_rate = max_charging_rate  # Max charging / discharging rate in kW
        self.efficiency = efficiency  # Charging/discharging efficiency
        self.battery_soc = init_soc * capacity  # Initial state of charge
        self.step_size = step_size

    def charge_discharge(self, power, duration, y_t):
        """
        Charges or discharges the battery with the given power for the specified duration.
        
        Parameters:
            power (Decimal): Power in kW to charge or discharge the battery.
            duration (Decimal): Duration in seconds for which the battery is charged/discharged.
            y_t (Decimal): The user load (kW) at the current step, used to determine how much the battery should discharge.

        Returns:
            Decimal: The power charged/discharged in kW, with efficiency considered.
        """

        assert type(power) is Decimal, "Power must be a Decimal."
        assert type(duration) is Decimal, "Duration must be a Decimal."
        assert type(y_t) is Decimal, "User load must be a Decimal."

        energy = power * (duration / Decimal(3600))

        if power >= 0:  # Charging
            return self.charge(energy, duration)
        else:
            return self.discharge(energy, duration, y_t)

    def charge(self, energy, duration):
        """
        Charge the battery with the given energy (kWh) for the specific duration (sec).

        This returned the consumed power in kW.

        Parameters:
            energy (Decimal): Energy in kWh to charge the battery.
            duration (Decimal): Duration in seconds for which the battery is charged.
        """

        assert type(energy) is Decimal, "Energy must be a Decimal."
        assert type(duration) is Decimal, "Duration must be a Decimal."

        energy_to_be_charged = energy * self.efficiency         # energy to be charged to battery soc
        energy_consumed = energy                                # energy consumed from the grid

        if self.battery_soc >= self.capacity:
            # Battery is already full, no energy can be charged
            energy_consumed = Decimal(0)
            return energy_consumed / (duration / Decimal(3600))

        
        if self.battery_soc + energy_consumed > self.capacity:
            energy_to_be_charged = self.capacity - self.battery_soc     # Only charge up to the capacity
            energy_consumed = energy_to_be_charged / self.efficiency    # energy_consumed is the lower limit drawn from the grid

            power_consumed = energy_consumed / (duration / Decimal(3600))  # power to be charged in kW
            # perform quantization on power_consumed, 
            # such that it is a multiple of step_size; 
            # and >= _power_consumed in above (if considering efficiency)
            # this is equivalent to a slightly decreased efficiency.
            power_consumed = (power_consumed / self.step_size).to_integral_exact(rounding=ROUND_CEILING) * self.step_size

            # perform quantization on energy_to_be_charged, such that it is a multiple of step_size
            energy_consumed = Decimal(power_consumed) * (duration / Decimal(3600))
            # energy_to_be_charged = energy_consumed * self.efficiency  # recalculate energy_to_be_charged based on the quantized power

            # Update the battery state of charge
            self.battery_soc += energy_to_be_charged
        else:
            self.battery_soc += energy_to_be_charged

        return energy_consumed / (duration / Decimal(3600))  # Return energy in kW

    def discharge(self, energy, duration, y_t):
        """
        Discharges the battery with the given energy (kWh) for the specified duration (sec).

        Parameters:
            energy (Decimal): Energy in kWh to discharge the battery.
            duration (Decimal): Duration in seconds for which the battery is discharged.
            y_t (Decimal): The user load at the current step, used to determine how much the battery should discharge.
        """

        assert type(energy) is Decimal, "Energy must be a Decimal."
        assert type(duration) is Decimal, "Duration must be a Decimal."
        assert type(y_t) is Decimal, "User load must be a Decimal."

        requested_energy_draw = -y_t * (duration / Decimal(3600)) / Decimal(1000)        # kWh

        energy_draw_from_battery = energy                                                                                   # energy to be deduced from battery soc
        # perform quantization on energy_discharged based on power quantization, 
        # such that the output power is a multiple of step_size; 
        # and <= energy_discharged (when considering efficiency)
        energy_discharged = int(energy * self.efficiency / (duration / Decimal(3600))  / self.step_size) * self.step_size  * (duration / Decimal(3600))

        if abs(energy_discharged) > abs(requested_energy_draw):
            energy_discharged = requested_energy_draw
            energy_draw_from_battery = requested_energy_draw / self.efficiency
        
        new_battery_soc = self.battery_soc + energy_draw_from_battery

        if new_battery_soc < 0:
            energy_discharged = -(self.battery_soc * self.efficiency)
            power_discharged = energy_discharged / (duration / Decimal(3600))                        # power discharged in kW
            # quantize the power discharged to the nearest step size
            # note that for efficiency < 1, this calculation will result in a power_discharged (below)
            # that is less than the power_discharged (above) due to the quantization.
            # this is equivalent to a slightly decreased efficiency.
            power_discharged = int(power_discharged / self.step_size) * self.step_size      # use integer to drop any decimal places
            energy_discharged = power_discharged * (duration / Decimal(3600))                        # recalculate energy discharged based on the quantized power

        self.battery_soc = max(new_battery_soc, Decimal(0))
        return energy_discharged / (duration / Decimal(3600))  # Return power in kW


    def compute_unnormalized_charge(self, normalized_action:Decimal):
        """
        Computes the unnormalized charge based on the normalized action.
        The normalized action is expected to be in the range [-1, 1].
        
        Parameters:
            normalized_action (Decimal): Normalized action value.
        
        Returns:
            Decimal: Unnormalized charge in kW.
        """
        if normalized_action < 0:
            # Discharging
            unnormalized_charge = normalized_action * self.max_charging_rate
        else:
            # Charging
            unnormalized_charge = normalized_action * self.max_charging_rate

        return unnormalized_charge

    def get_normalized_state_of_charge(self):
        """
        Returns the normalized [0, 1] current state of charge (SoC) of the battery.
        """
        return self.battery_soc / self.capacity

    def reset(self, init_soc=0.5):
        """
        Resets the battery to a specified initial state of charge.
        Parameters:
            init_soc (float): Initial state of charge as a fraction of capacity (default is 0.5).
        """
        self.battery_soc = init_soc * self.capacity

    def get_battery_config(self):
        """
        Returns the configuration of the battery as a dictionary.
        """
        return {
            'capacity': float(self.capacity),
            'max_charging_rate': float(self.max_charging_rate),
            'efficiency': float(self.efficiency),
            'initial_soc': float(self.battery_soc / self.capacity)
        }