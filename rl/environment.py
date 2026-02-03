import random
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from grammar.masker import GrammarMasker
from grammar.state import GrammarState
from grammar.parser import render_sequence, render_model
import grammar.vocab as vocab
from config import config
from reward.reward import all_rewards
from rl.curriculum import Curriculum


class TheoryEnvironment:
    def __init__(self, cfg=config, curriculum: Curriculum = None):
        self.config = cfg
        self.curriculum = curriculum
        self.grammar_masker = GrammarMasker()
    
        # Episode state
        self.state: Optional[GrammarState] = None
        self.sequence: List[str] = []
        self.valid_token_history: List[List[int]] = []
        self.prev_token: str = "BOS"
        self.done: bool = False
        self.theory_too_long: bool = False
        self.step_count: int = 0
        self.total_score: float = 0
        self.rewards: dict = {}
        self.termination_reason: str = "ongoing"
        
        # Statistics
        self.episode_count: int = 0
        self.total_steps: int = 0
        
    def reset(self, seed: Optional[int] = None, prompt: Optional[List[str]] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset episode state
        self.state = self.grammar_masker.init_state()
        self.sequence = ["BOS"]
        self.valid_token_history = [[0]]  # BOS encoded as 0
        self.prev_token = "BOS"
        self.done = False
        self.theory_too_long = False
        self.step_count = 0
        self.episode_count += 1
        self.total_score = 0
        self.rewards = {}
        self.termination_reason = "ongoing"

        # Apply prompt if provided
        if prompt:
            if prompt[0] == "BOS": prompt = prompt[1:]
            if prompt[-1] == "EOS": prompt = prompt[:-1]
            
            for token in prompt:
                valid_tokens = self.grammar_masker.get_valid_tokens(self.state, self.prev_token)
                # Safety check
                if valid_tokens is None:
                    print(f"ERROR: get_valid_tokens returned None for token='{self.prev_token}', state.current_block='{self.state.current_block}'")
                    valid_tokens = ["EOS"]  # Fallback
                self.sequence.append(token)
                self.valid_token_history.append(vocab.encode(valid_tokens))
                self.state = self.grammar_masker.step(self.state, token)
                self.prev_token = token
                self.step_count += 1
        
        return self._get_observation()
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        
        # Get valid tokens
        valid_tokens = self.grammar_masker.get_valid_tokens(self.state, self.prev_token)
        
        # Safety check
        if valid_tokens is None:
            print(f"ERROR: get_valid_tokens returned None for token='{self.prev_token}', state.current_block='{self.state.current_block}'")
            valid_tokens = ["EOS"]  # Fallback
        
        # Execute action
        self.sequence.append(action)
        self.valid_token_history.append(vocab.encode(valid_tokens))
        self.state = self.grammar_masker.step(self.state, action)
        self.prev_token = action
        self.step_count += 1
        self.total_steps += 1
        
        # Check termination conditions
        self._check_termination(action)

        # Calculate reward if episode is done
        if self.done: 
            self.total_score = self._calculate_reward()
        
        # Prepare info dictionary
        info = {
            "success": self.done and not self.theory_too_long,
            "termination_reason": self._get_termination_reason(),
            "step_count": self.step_count,
        }
        
        # Add reward details if done
        if self.done:
            info.update(self.rewards)
        
        # Get observation
        observation = self._get_observation()
        
        # Return (observation, reward, done, info)
        # Reward is 0 for intermediate steps, total_score when done
        reward = self.total_score if self.done else 0.0
        
        return observation, reward, self.done, info
            


    def _get_observation(self) -> Dict[str, Any]:
        valid_tokens = self.grammar_masker.get_valid_tokens(self.state, self.prev_token)
        
        # Handle terminal state where there are no more valid tokens
        if valid_tokens is None: valid_tokens = []
        
        return {"sequence": self.sequence.copy(),
                "prev_token": self.prev_token,
                "valid_tokens": valid_tokens,
                "valid_token_ids": vocab.encode(valid_tokens),
                "step_count": self.step_count
                }
    
    def _check_termination(self, action: str):
        # Check for EOS token
        if action == "EOS":
            self.done = True
            self.termination_reason = "eos"
            
        # Check for length limit
        elif action in ["THEORY_TOO_LONG", "TOO_MANY_INTERACTIONS", "TOO_MANY_PARAMS"]:
            self.done = True
            self.theory_too_long = True
            self.termination_reason = action.lower()
            
        # Check max length
        elif self.state.length >= self.config.MAX_LENGTH - 1:
            self.done = True
            self.theory_too_long = True
            self.termination_reason = "max_length_reached"
    
    def _get_termination_reason(self) -> str:
        """Get the reason for episode termination."""
        if not self.done:
            return "ongoing"
        return getattr(self, "termination_reason", "unknown")
    
    def _calculate_reward(self) -> float:
        assert self.done, "Episode must be done to calculate reward"
        
        self.rewards = all_rewards(self._get_full_model())
        if self.curriculum: 
            total_reward = self.curriculum.calculate_total_reward(self.rewards)
        else:
            total_reward = self.rewards["total_reward"]
        return total_reward
    
    def _get_full_model(self) -> Dict[str, Any]:
        return {"too_long": self.theory_too_long,
                "gauge_groups": self.state.gauge_groups,
                "vevs": self.state.vevs,
                "particles": self.state.particles,
                "multiplets": self.state.multiplets,
                "interactions": self.state.interactions,
                "anomalies": self.state.anomalies,
                "params": self.state.params,
                "sequence": self.sequence,
                "valid_tokens": self.valid_token_history
                }
    
    def render(self, mode: str = "human", full_model: bool = False) -> Optional[str]:
        if mode == "human":
            if full_model: return render_model(self._get_full_model())
            else: return render_sequence(self.sequence)
        return None
    

if __name__ == "__main__":
    from grammar.parser import read_model_txt

    length_rewards = []
    light_exotics_rewards = []
    anomaly_rewards = []

    prompt = read_model_txt("dataset/prompts/SM_gauge_prompt.txt")
    curriculum = Curriculum()
    curriculum.phase = 1
    env = TheoryEnvironment(curriculum=curriculum)
    for i in range(5000):
        env.reset(prompt=prompt)
        while not env.done:
            observation = env._get_observation()
            valid_tokens = observation["valid_tokens"]
            action = random.choice(valid_tokens)
            env.step(action)
            env._check_termination(action)

        length_rewards.append(env.rewards["length_reward"])
        light_exotics_rewards.append(env.rewards["light_exotics_reward"])
        anomaly_rewards.append(env.rewards["anomalies_reward"])

    #print(length_rewards)
    print(f"Length Reward: {np.mean(length_rewards):.2f}, {np.std(length_rewards):.2f}")
    print(f"Min Length Reward: {np.min(length_rewards):.2f}, Max Length Reward: {np.max(length_rewards):.2f}")
    #print(light_exotics_rewards)
    print(f"Light Exotics Reward: {np.mean(light_exotics_rewards):.2f}, {np.std(light_exotics_rewards):.2f}")
    print(f"Min Light Exotics Reward: {np.min(light_exotics_rewards):.2f}, Max Light Exotics Reward: {np.max(light_exotics_rewards):.2f}")
    #print(anomaly_rewards)
    print(f"Anomaly Reward: {np.mean(anomaly_rewards):.2f}, {np.std(anomaly_rewards):.2f}")
    print(f"Min Anomaly Reward: {np.min(anomaly_rewards):.2f}, Max Anomaly Reward: {np.max(anomaly_rewards):.2f}")
