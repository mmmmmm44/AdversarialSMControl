from enum import Enum

class HNetworkType(Enum):
    """
    Enum to define the type of H-Network.
    """
    H_NETWORK = "h_network"  # H-Network that predicts only the mean
    H_NETWORK2 = "h_network2"  # H-Network that predicts both mean and log variance