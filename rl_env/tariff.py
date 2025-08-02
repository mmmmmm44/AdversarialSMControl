# defining constants for the tariff of time-of-use pricing
# unitL: GBP/kWh

from datetime import time

# Define the time-of-use pricing structure
TIME_OF_USE_PRICES = {
    (time(0,0,0,0), time(7,0,0,0)): 0.1317,              # off-peak price
    (time(7,0,0,0), time(23,59,59,999999)): 0.3075,      # peak price
}

STANDING_CHARGE = 0.4734  # in GBP