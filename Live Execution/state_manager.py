import json
import os

class StateManager:
    def __init__(self, filename='bot_state.json'):
        self.filename = filename
        self.state = {}
        self.load_state()

    def load_state(self):
        """
        Reads the state from the file. 
        If file doesn't exist, creates a blank one.
        """
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.state = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start fresh
                print("Warning: State file corrupted. Starting fresh.")
                self.state = {}
        else:
            # File doesn't exist yet, start empty
            self.state = {}

    def save_state(self):
        """Writes the current memory to the hard drive."""
        with open(self.filename, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_strategy_state(self, strategy_name):
        """
        Returns the dict for a specific strategy.
        If it doesn't exist, initializes it as 'FLAT' (no trade).
        """
        if strategy_name not in self.state:
            self.state[strategy_name] = {
                "status": "FLAT",
                "entry_price": 0.0,
                "amount": 0.0,
                "entry_time": None
            }
            self.save_state()
            
        return self.state[strategy_name]

    def update_position(self, strategy_name, status, price=0.0, amount=0.0):
        """
        Updates the state when you Buy or Sell.
        status: 'IN_POSITION' or 'FLAT'
        """
        from datetime import datetime
        
        self.state[strategy_name] = {
            "status": status,
            "entry_price": price,
            "amount": amount,
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status == 'IN_POSITION' else None
        }
        self.save_state()
        print(f"State updated for {strategy_name}: {status}")