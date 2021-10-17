import json
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

import seaborn as sns

# Apply the default theme
sns.set_theme()

def dreamerv2_results(file_path):
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    dreamerv2_steps = []
    dreamerv2_eval_returns = []
    for json_str in json_list:
        result = json.loads(json_str)
        if 'eval_return' in result.keys():
            dreamerv2_steps.append(result['step'])
            dreamerv2_eval_returns.append(result['eval_return'])
            #print(f"result: {result}")
            #print(isinstance(result, dict))
    return (np.array(dreamerv2_steps), np.array(dreamerv2_eval_returns))

def new_model_results(file_path):
    data = torch.load(file_path)
    steps = np.array(data['steps'])[109::10]
    test_rewards = np.squeeze(np.array(data['test_rewards']))
    return (steps, test_rewards)

def planet_results(file_path):
    data = torch.load(file_path)
    steps = np.array(data['steps'])[24::25]
    test_rewards = np.squeeze(np.array(data['test_rewards']))
    return (steps, test_rewards)

env = "humanoid"

dreamerv2_steps_seed_1, dreamerv2_eval_returns_seed1 = dreamerv2_results(f"dreamerv2/{env}/{env}/metrics_1.jsonl")
dreamerv2_steps_seed_2, dreamerv2_eval_returns_seed2 = dreamerv2_results(f"dreamerv2/{env}/{env}/metrics_2.jsonl")
dreamerv2_steps_seed_3, dreamerv2_eval_returns_seed3 = dreamerv2_results(f"dreamerv2/{env}/{env}/metrics_3.jsonl")
num_dreamerv2samples = min(len(dreamerv2_steps_seed_1), len(dreamerv2_steps_seed_2), len(dreamerv2_steps_seed_3))
dreamerv2_average_eval_return = (dreamerv2_eval_returns_seed1[:num_dreamerv2samples] + dreamerv2_eval_returns_seed2[:num_dreamerv2samples] + dreamerv2_eval_returns_seed3[:num_dreamerv2samples])/3

regular_vae_steps_seed1, regular_vae_test_rewards_seed1 = new_model_results(f"regular_vae/{env}/1/metrics.pth")
regular_vae_steps_seed2, regular_vae_test_rewards_seed2 = new_model_results(f"regular_vae/{env}/2/metrics.pth")
regular_vae_steps_seed3, regular_vae_test_rewards_seed3 = new_model_results(f"regular_vae/{env}/3/metrics.pth")
num_regular_vae_samples = min(len(regular_vae_steps_seed1), len(regular_vae_steps_seed2), len(regular_vae_steps_seed3))
regular_vae_average_eval_return = (regular_vae_test_rewards_seed1[:num_regular_vae_samples] + regular_vae_test_rewards_seed2[:num_regular_vae_samples] + regular_vae_test_rewards_seed3[:num_regular_vae_samples])/3

planet_steps_seed1, planet_test_rewards_seed1 = planet_results(f"planet/{env}/1/metrics.pth")
planet_steps_seed2, planet_test_rewards_seed2 = planet_results(f"planet/{env}/2/metrics.pth")
planet_steps_seed3, planet_test_rewards_seed3 = planet_results(f"planet/{env}/3/metrics.pth")
num_planet_samples = min(len(planet_steps_seed1), len(planet_steps_seed2), len(planet_steps_seed3))
planet_average_eval_return = (planet_test_rewards_seed1[:num_planet_samples] + planet_test_rewards_seed2[:num_planet_samples] + planet_test_rewards_seed3[:num_planet_samples])/3

'''
infinite_vae_steps_seed1, infinite_vae_test_rewards_seed1 = new_model_results(f"infinite_vae/{env}/1/metrics.pth")
infinite_vae_steps_seed2, infinite_vae_test_rewards_seed2 = new_model_results(f"infinite_vae/{env}/2/metrics.pth")
infinite_vae_steps_seed3, infinite_vae_test_rewards_seed3 = new_model_results(f"infinite_vae/{env}/3/metrics.pth")
num_infinite_vae_samples = min(len(infinite_vae_steps_seed1), len(infinite_vae_steps_seed2), len(infinite_vae_steps_seed3))
infinite_vae_average_eval_return = (infinite_vae_test_rewards_seed1[:num_infinite_vae_samples] + infinite_vae_test_rewards_seed2[:num_infinite_vae_samples] + infinite_vae_test_rewards_seed3[:num_infinite_vae_samples])/3
'''

plt.plot(regular_vae_steps_seed1[:num_regular_vae_samples], regular_vae_average_eval_return, label="D2E")
plt.plot(dreamerv2_steps_seed_1[:num_dreamerv2samples], dreamerv2_average_eval_return, label="Dreamerv2")
plt.plot(planet_steps_seed1[:num_planet_samples], planet_average_eval_return, label="Planet")
#plt.plot(infinite_vae_test_rewards_seed1[:num_infinite_vae_samples], infinite_vae_average_eval_return, label="infinite vae model")

plt.legend()
plt.title(f"{env.replace('_', ' ').capitalize()}")
plt.xlabel("Steps")
plt.ylabel("Average Test Return/Reward")
plt.ticklabel_format(axis='x', style='sci', scilimits=(4,4))

plt.savefig(f"final_plots/final_plot_{env}.png")