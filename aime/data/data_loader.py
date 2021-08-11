import gym
from envs import register_envs
import torch

register_envs()

def generate_data(env_name, mode=None, num_random_episodes=1000, num_lagging_observation=2, num_lagging_action=1):
    env = gym.make(env_name)
    episode_list = []
    num_current_episodes = 0
    while num_current_episodes < num_random_episodes:
        obs_list = []
        reward_list = []
        image_list = []
        action_list = []
        obs = env.reset()
        image = env.render(mode="rgb_array")
        obs_list.append(obs)
        image_list.append(image)
        done = False
        num_steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            image = env.render(mode="rgb_array")
            action_list.append(action)
            obs_list.append(obs)
            reward_list.append(reward)
            image_list.append(image)
            num_steps += 1
        if (num_steps >= num_lagging_action) and (num_steps+1 >= num_lagging_observation):
            print(num_steps)
            episode_data = {
                "observation_list": obs_list,
                "reward_list": reward_list,
                "image_list": image_list,
                "action_list": action_list
            }
            episode_list.append(episode_data)
            num_current_episodes += 1
    # save episode data to pickle file
    # episode_list.to_pickle()
    env.close()
    return episode_list

def preprocess_data(data):
    observation_tensor = torch.Tensor([d["observation_list"] for d in data])
    image_tensor = torch.Tensor([d["image_list"] for d in data])
    action_tensor = torch.Tensor([d["action_list"] for d in data])
    reward_tensor = torch.Tensor([d["reward_list"] for d in data])
    preprocessed_data = {
        "observation_tensor": observation_tensor,
        "image_tensor": image_tensor,
        "action_tensor": action_tensor,
        "reward_tensor": reward_tensor
    }
    return preprocessed_data
    
def load_data(env_name, mode=None, num_random_episodes=1000, num_lagging_observation=2, num_lagging_action=1):
    data = generate_data(env_name, mode, num_random_episodes, num_lagging_observation, num_lagging_action)
    return preprocess_data(data)

if __name__ == '__main__':
    episode_list = load_data('MountainCarContinuous-v0', num_random_episodes=4)
    print(episode_list)

