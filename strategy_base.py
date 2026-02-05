import pandas as pd

# --- 1. THE TEMPLATE (Base Class) ---
class StrategyBase:
    def __init__(self, name):
        self.name = name
    
    def on_data(self, row):
        """
        Input: A single row of data (timestamp, price, z_score, etc.)
        Output: Target Position % (0.0 to 1.0)
        """
        raise NotImplementedError
    
    def get_name(self):
        "Return a string with the name of the strategy"
        raise NotImplementedError