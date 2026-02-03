"""
Simple plot of reward and loss from training log.
"""

import re
import matplotlib.pyplot as plt

# Parse log file
log_file = "log/error_221869.txt"
episodes = []
rewards = []
losses = []

pattern = r'\|\s*(\d+)/\d+\s*\[.*?reward=([-\d.]+).*?loss=([-\d.]+)'

with open(log_file, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            loss = float(match.group(3))
            
            if not episodes or episode != episodes[-1]:
                episodes.append(episode)
                rewards.append(reward)
                losses.append(loss)

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Reward plot
ax1.plot(episodes[:61], rewards[:61], 'b-', linewidth=1)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Reward')
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(episodes[:61], losses[:61], 'r-', linewidth=1)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('log/training_simple.png', dpi=150)
print(f"Saved plot to: log/training_simple.png")
print(f"Episodes: {len(episodes)}")
print(f"Best reward: {max(rewards):.2f} (episode {episodes[rewards.index(max(rewards))]})")
print(f"Worst reward: {min(rewards):.2f} (episode {episodes[rewards.index(min(rewards))]})")
