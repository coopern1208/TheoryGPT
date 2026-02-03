from grammar.gen_theory import generate_theory
from grammar.parser import read_model_txt, render_sequence, render_model
from grammar.masker import GrammarMasker

import reward.reward as reward
import random
import numpy as np
import matplotlib.pyplot as plt

prompt = read_model_txt("dataset/prompts/SM_full.txt")
all_rewards = []
perfect_case = 0

for i in range(1):
    grammar_masker = GrammarMasker()
    full_model = generate_theory(grammar_masker, 
                                 prompt=prompt, 
                                 print_tokens=False, 
                                 seed=None)

    
    rewards = reward.all_rewards(full_model)

    if rewards['total_reward'] > -1: perfect_case += 1
    all_rewards.append(rewards['total_reward'])
    print("=============================== Rewards Seed =============================== \n")
    print(f"Anomalies Reward    : {rewards['anomalies_reward']:.2f}")
    print(f"Light Exotics       : {rewards['light_exotics_reward']:.2f}")
    print(f"TOTAL REWARD        : {rewards['total_reward']:.2f}\n")

    print(full_model)

# print(f"Perfect Case        : {perfect_case/1000:.2f}")
# print(np.mean(all_rewards))
# plt.hist(all_rewards, bins=50)
# plt.savefig("reward/reward_distribution_1.png")
