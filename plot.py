import json
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import pandas as pd

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
    test_rewards = np.mean(np.array(data['test_rewards']), axis=1)
    return (steps, test_rewards)

def slac_results(file_path):
    data = pd.read_csv(file_path)
    steps = data['step']
    returns = data['return']
    return (steps, returns)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PlaNet')
    parser.add_argument('--env', type=str, choices=['cheetah-run', 'hopper-hop', 'humanoid-walk', 'swimmer-swimmer6', 'walker-walk'], help='Environment Name')
    args = parser.parse_args()
    env = args.env

    dreamerv2_steps_seed_1, dreamerv2_eval_returns_seed1 = dreamerv2_results(f"dreamerv2/dreamerv2_dmc/{env}/run0.jsonl")
    dreamerv2_steps_seed_2, dreamerv2_eval_returns_seed2 = dreamerv2_results(f"dreamerv2/dreamerv2_dmc/{env}/run1.jsonl")
    dreamerv2_steps_seed_3, dreamerv2_eval_returns_seed3 = dreamerv2_results(f"dreamerv2/dreamerv2_dmc/{env}/run2.jsonl")
    num_dreamerv2samples = min(len(dreamerv2_steps_seed_1), len(dreamerv2_steps_seed_2), len(dreamerv2_steps_seed_3))
    dreamerv2_steps_seed_1 = dreamerv2_steps_seed_1[:num_dreamerv2samples]
    dreamerv2_steps_seed_2 = dreamerv2_steps_seed_2[:num_dreamerv2samples]
    dreamerv2_steps_seed_3 = dreamerv2_steps_seed_3[:num_dreamerv2samples]
    dreamerv2_eval_returns_seed1 = dreamerv2_eval_returns_seed1[:num_dreamerv2samples]
    dreamerv2_eval_returns_seed2 = dreamerv2_eval_returns_seed2[:num_dreamerv2samples]
    dreamerv2_eval_returns_seed3 = dreamerv2_eval_returns_seed3[:num_dreamerv2samples]
    dreamerv2_rewards = np.stack([dreamerv2_eval_returns_seed1, dreamerv2_eval_returns_seed2, dreamerv2_eval_returns_seed3], axis=1)
    dreamerv2_average_eval_return = np.mean(dreamerv2_rewards, axis=1)
    dreamerv2_std = np.std(dreamerv2_rewards, axis=1)
    dreamerv2_average_steps = (dreamerv2_steps_seed_1 + dreamerv2_steps_seed_2 + dreamerv2_steps_seed_3)/3

    slac_steps_seed_1, slac_returns_seed_1 = slac_results(f"slac/{env}/1/log.csv")
    slac_steps_seed_2, slac_returns_seed_2 = slac_results(f"slac/{env}/2/log.csv")
    slac_steps_seed_3, slac_returns_seed_3 = slac_results(f"slac/{env}/3/log.csv")
    num_slac_samples = min(len(slac_steps_seed_1), len(slac_steps_seed_2), len(slac_steps_seed_3))
    slac_steps_seed_1 = slac_steps_seed_1[:num_slac_samples]
    slac_steps_seed_2 = slac_steps_seed_2[:num_slac_samples]
    slac_steps_seed_3 = slac_steps_seed_3[:num_slac_samples]
    slac_returns_seed_1 = slac_returns_seed_1[:num_slac_samples]
    slac_returns_seed_2 = slac_returns_seed_2[:num_slac_samples]
    slac_returns_seed_3 = slac_returns_seed_3[:num_slac_samples]
    slac_rewards = np.stack([slac_returns_seed_1, slac_returns_seed_2, slac_returns_seed_3], axis=1)
    slac_average_return = np.mean(slac_rewards, axis=1)
    slac_std = np.std(slac_rewards, axis=1)
    slac_average_steps = (slac_steps_seed_1 + slac_steps_seed_2 + slac_steps_seed_3) / 3

    planet_steps_seed1, planet_test_rewards_seed1 = planet_results(f"planet/{env}/1/metrics.pth")
    planet_steps_seed2, planet_test_rewards_seed2 = planet_results(f"planet/{env}/2/metrics.pth")
    planet_steps_seed3, planet_test_rewards_seed3 = planet_results(f"planet/{env}/3/metrics.pth")
    num_planet_samples = min(len(planet_steps_seed1), len(planet_steps_seed2), len(planet_steps_seed3))
    planet_steps_seed1 = planet_steps_seed1[:num_planet_samples]
    planet_steps_seed2 = planet_steps_seed2[:num_planet_samples]
    planet_steps_seed3 = planet_steps_seed3[:num_planet_samples]
    planet_test_rewards_seed1 = planet_test_rewards_seed1[:num_planet_samples]
    planet_test_rewards_seed2 = planet_test_rewards_seed2[:num_planet_samples]
    planet_test_rewards_seed3 = planet_test_rewards_seed3[:num_planet_samples]
    planet_rewards = np.stack([planet_test_rewards_seed1, planet_test_rewards_seed2, planet_test_rewards_seed3], axis=1)
    planet_average_eval_return = np.mean(planet_rewards, axis=1)
    planet_std = np.std(planet_rewards, axis=1)
    planet_average_steps = (planet_steps_seed1 + planet_steps_seed2 + planet_steps_seed3)/3
    
    regular_vae_steps_seed1, regular_vae_test_rewards_seed1 = new_model_results(f"regular_vae/{env}/1/metrics.pth")
    regular_vae_steps_seed2, regular_vae_test_rewards_seed2 = new_model_results(f"regular_vae/{env}/2/metrics.pth")
    regular_vae_steps_seed3, regular_vae_test_rewards_seed3 = new_model_results(f"regular_vae/{env}/3/metrics.pth")
    num_regular_vae_samples = min(len(regular_vae_steps_seed1), len(regular_vae_steps_seed2), len(regular_vae_steps_seed3))
    regular_vae_steps_seed1 = regular_vae_steps_seed1[:num_regular_vae_samples]
    regular_vae_steps_seed2 = regular_vae_steps_seed2[:num_regular_vae_samples]
    regular_vae_steps_seed3 = regular_vae_steps_seed3[:num_regular_vae_samples]
    regular_vae_test_rewards_seed1 = regular_vae_test_rewards_seed1[:num_regular_vae_samples]
    regular_vae_test_rewards_seed2 = regular_vae_test_rewards_seed2[:num_regular_vae_samples]
    regular_vae_test_rewards_seed3 = regular_vae_test_rewards_seed3[:num_regular_vae_samples]
    regular_vae_rewards = np.stack([regular_vae_test_rewards_seed1, regular_vae_test_rewards_seed2, regular_vae_test_rewards_seed3], axis=1)
    regular_vae_average_eval_return = np.mean(regular_vae_rewards, axis=1)
    regular_vae_std = np.std(regular_vae_rewards, axis=1)
    regular_vae_average_steps = (regular_vae_steps_seed1 + regular_vae_steps_seed2 + regular_vae_steps_seed3)/3

    '''
    infinite_vae_steps_seed1, infinite_vae_test_rewards_seed1 = new_model_results(f"infinite_vae/{env}/1/metrics.pth")
    infinite_vae_steps_seed2, infinite_vae_test_rewards_seed2 = new_model_results(f"infinite_vae/{env}/2/metrics.pth")
    infinite_vae_steps_seed3, infinite_vae_test_rewards_seed3 = new_model_results(f"infinite_vae/{env}/3/metrics.pth")
    num_infinite_vae_samples = min(len(infinite_vae_steps_seed1), len(infinite_vae_steps_seed2), len(infinite_vae_steps_seed3))
    infinite_vae_average_eval_return = (infinite_vae_test_rewards_seed1[:num_infinite_vae_samples] + infinite_vae_test_rewards_seed2[:num_infinite_vae_samples] + infinite_vae_test_rewards_seed3[:num_infinite_vae_samples])/3
    infinite_vae_average_steps = (infinite_vae_steps_seed1[:num_infinite_vae_samples] + infinite_vae_steps_seed2[:num_infinite_vae_samples] + infinite_vae_steps_seed3[:num_infinite_vae_samples])/3
    '''

    plt.plot(dreamerv2_average_steps, dreamerv2_average_eval_return, label="Dreamerv2", color="red")
    plt.fill_between(dreamerv2_average_steps, dreamerv2_average_eval_return-dreamerv2_std, dreamerv2_average_eval_return+dreamerv2_std, color="red", alpha=0.2)
    plt.plot(slac_average_steps, slac_average_return, label="Slac", color="green")
    plt.fill_between(slac_average_steps, slac_average_return-slac_std, slac_average_return+slac_std, color="green", alpha=0.2)
    plt.plot(planet_average_steps, planet_average_eval_return, label="Planet", color='blue')
    plt.fill_between(planet_average_steps, planet_average_eval_return-planet_std, planet_average_eval_return+planet_std, color='blue', alpha=0.2)
    plt.plot(regular_vae_average_steps, regular_vae_average_eval_return, label="D2E (Regular VAE)", color="black")
    plt.fill_between(regular_vae_average_steps, regular_vae_average_eval_return-regular_vae_std, regular_vae_average_eval_return+regular_vae_std, color='black', alpha=0.2)
    #plt.plot(infinite_vae_average_steps, infinite_vae_average_eval_return, label="D2E (Infinite Mixture VAE)")

    plt.legend()
    plt.title(f"{env.replace('_', ' ').capitalize()}")
    plt.xlabel("Steps")
    plt.ylabel("Average Test Return/Reward")
    plt.ticklabel_format(axis='x', style='sci', scilimits=(4,4))

    plt.savefig(f"final_plots/final_plot_{env}.png")