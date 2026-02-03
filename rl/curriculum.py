from typing import Dict
import torch
from collections import deque

class Curriculum:
    def __init__(self, reward_threshold: float = 0.0, mastery_density: float = 0.9, max_history_length: int = 100):
        self.phase = 3
        self.reward_threshold = reward_threshold
        self.mastery_density = mastery_density
        self.history = deque(maxlen=max_history_length)
        
    def calculate_total_reward(self, rewards: Dict[str, float]) -> float:
        if self.phase == 1: 
            return rewards['length_reward']
        elif self.phase == 2:
            return rewards['length_reward'] + rewards['light_exotics_reward']
        elif self.phase == 3:
            return rewards['length_reward'] + rewards['light_exotics_reward'] + rewards['anomalies_reward']
    
    def update_phase(self, current_batch_rewards):
        batch_perfect_rate = (current_batch_rewards >= self.reward_threshold).float().mean().item()
        self.history.append(batch_perfect_rate)

        avg_perfect_rate = sum(self.history) / len(self.history)

        if len(self.history) == self.history.maxlen and avg_perfect_rate >= self.mastery_density:
            if self.phase < 3:
                self.phase += 1
                self.history.clear()
                print(f"--- PHASE {self.phase} UNLOCKED ---")