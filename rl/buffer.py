import random
import pickle
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from config import config
import torch

class RLBuffer:
    """Buffer for storing successful RL trajectories"""
    
    def __init__(self, device: torch.device = torch.device("cpu"), max_size: int = config.REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        self.device = device
        self.trajectories = {
            1: deque[Any](maxlen=max_size),
            2: deque[Any](maxlen=max_size),
            3: deque[Any](maxlen=max_size)
        }
        
    def add(self, sequence: List[str], phase: int, is_success: bool):
        if is_success:
            trajectory = torch.tensor(sequence, dtype=torch.long).to(self.device)
            self.trajectories[phase].append(trajectory)
    
    def sample(self, batch_size: int, phase: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        batch_size = min(batch_size, len(self.trajectories[phase]))
        samples = []

        if phase == 1:
            pass

        indices = random.sample(range(len(self.trajectories)), batch_size)
        sampled_trajectories = [self.trajectories[i] for i in indices]
        
        return sampled_trajectories
    
    
    def get_statistics(self, phase: int = 1) -> Dict[str, float]:
        if phase in [1, 2, 3]:
            return {"count": len(self.trajectories[phase]), "success_rate": len(self.trajectories[phase]) / self.max_size}
        else:
            return {1: {"count": len(self.trajectories[1]), "success_rate": len(self.trajectories[1]) / self.max_size},
                    2: {"count": len(self.trajectories[2]), "success_rate": len(self.trajectories[2]) / self.max_size},
                    3: {"count": len(self.trajectories[3]), "success_rate": len(self.trajectories[3]) / self.max_size}}

    def _get_from_buffer(self, phase: int, num: int) -> List[Dict[str, Any]]:
        if len(self.trajectories[phase]) < num: 
            return self.trajectories[phase]
        else:
            return random.sample(self.trajectories[phase], num)
      
    def save(self, filepath: str):
        data = {
            "max_size": self.max_size,
            "trajectories": list(self.trajectories),
            "rewards": list(self.rewards)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load the buffer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.max_size = data["max_size"]
        self.trajectories = deque[Any](data["trajectories"], maxlen=self.max_size)



if __name__ == "__main__":
    print("=" * 50)
    print("Testing ReplayBuffer")
    print("=" * 50)
    
    trajectory = torch.randint(0, 200, (10, 100))
    reward = torch.rand(10)

