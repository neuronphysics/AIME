import gym
from envs import register_envs

register_envs()

def load_data(env_name, mode=None, num_random_episodes=1000, num_lagging_observation=2, num_lagging_action=1):
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
    return episode_list

if __name__ == '__main__':
    episode_list = load_data('MountainCarContinuous-v0', num_random_episodes=10)
    print(episode_list)
    
    
            