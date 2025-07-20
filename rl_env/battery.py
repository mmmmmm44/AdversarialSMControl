"""
The rechargeable battery module simulates a rechargeable battery system.
It provides methods to charge, discharge, and check the battery's state of charge (SoC).
It also includes methods to calculate the energy stored in the battery and to reset its state.
"""


class RechargeableBattery:
    
    def __init__(self, capacity, max_charging_rate, max_discharging_rate, efficiency=1, init_soc=0.5):
        """
        Initializes the rechargeable battery with given parameters.
        Parameters:
            capacity (float): Total capacity of the battery in Wh.
            max_charging_rate (float): Maximum charging rate in kW.
            max_discharging_rate (float): Maximum discharging rate in kW.
            efficiency (float): Efficiency of charging/discharging (default is 1, i.e., 100%).
            init_soc (float): Initial state of charge (SoC) as a fraction of capacity (default is 0.5).
        """

        self.capacity = capacity  # Total capacity in kWh
        self.max_charging_rate = max_charging_rate  # Max charging rate in kW
        self.max_discharging_rate = max_discharging_rate  # Max discharging rate in kW
        self.efficiency = efficiency  # Charging/discharging efficiency
        self.battery_soc = init_soc * capacity  # Initial state of charge


    def charge_discharge(self, power, duration, y_t):
        """
        Charges or discharges the battery with the given power for the specified duration.
        
        Parameters:
            power (float): Power in kW to charge or dis0charge the battery.
            duration (float): Duration in seconds for which the battery is charged/discharged.
            y_t (float): The user load (kW) at the current step, used to determine how much the battery should discharge.
        
        Returns:
            float: The power charged/discharged in kW, with efficiency considered.
        """
        energy = power * (duration / 3600)

        if power >= 0:  # Charging
            return self.charge(energy, duration)
        else:
            return self.discharge(energy, duration, y_t)

    def charge(self, energy, duration):
        """
        Charge the battery with the given energy (kWh) for the specific duration (sec).

        This returned the consumed power in kWh.

        Parameters:
            power (float): Power in kW to charge the battery.
            duration (float): Duration in seconds for which the battery is charged.
        """

        energy_to_be_charged = energy * self.efficiency         # energy to be charged to battery soc
        energy_consumed = energy                                # energy consumed from the grid

        if self.battery_soc >= self.capacity:
            # Battery is already full, no energy can be charged
            energy_consumed = 0
            return energy_consumed / (duration / 3600)

        
        if self.battery_soc + energy_consumed > self.capacity:
            energy_to_be_charged = self.capacity - self.battery_soc  # Only charge up to capacity
            energy_consumed = energy_to_be_charged / self.efficiency
            
            self.battery_soc = self.capacity
        else:
            self.battery_soc += energy_to_be_charged

        return energy_consumed / (duration / 3600)  # Return energy in kW

    def discharge(self, energy, duration, y_t):
        """
        Discharges the battery with the given energy (kWh) for the specified duration (sec).

        Parameters:
            energy (float): Energy in kWh to discharge the battery.
            duration (float): Duration in seconds for which the battery is discharged.
            y_t (float): The user load at the current step, used to determine how much the battery should discharge.
        """

        requested_energy_draw = -y_t * (duration / 3600) / 1000        # kWh

        energy_draw_from_battery = energy                       # energy to be deduced from battery soc
        energy_discharged = energy * self.efficiency            # energy discharged to support user load

        if abs(energy_discharged) > abs(requested_energy_draw):
            energy_discharged = requested_energy_draw
            energy_draw_from_battery = requested_energy_draw / self.efficiency
        
        new_battery_soc = self.battery_soc + energy_draw_from_battery

        if new_battery_soc < 0:
            energy_discharged = -(self.battery_soc * self.efficiency)

        self.battery_soc = max(new_battery_soc, 0)
        return energy_discharged / (duration / 3600)  # Return energy in kW


    def compute_unnormalized_charge(self, normalized_action):
        """
        Computes the unnormalized charge based on the normalized action.
        The normalized action is expected to be in the range [-1, 1].
        
        Parameters:
            normalized_action (float): Normalized action value.
        
        Returns:
            float: Unnormalized charge in kW.
        """
        if normalized_action < 0:
            # Discharging
            unnormalized_charge = normalized_action * self.max_discharging_rate
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
            'capacity': self.capacity,
            'max_charging_rate': self.max_charging_rate,
            'max_discharging_rate': self.max_discharging_rate,
            'efficiency': self.efficiency,
            'initial_soc': self.battery_soc / self.capacity
        }