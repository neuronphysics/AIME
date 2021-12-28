from torchvision.transforms.functional import resize
import numpy as np
import torch
from gym.spaces import Box
import pybullet_envs

import dmc2gym
import gym


GYM_ENVS = ['HumanoidBulletEnv-v0', 'Pendulum-v0', 'MountainCarContinuous-v0', 'CarRacing-v0', 'CartPoleContinuousBulletEnv-v0', 'AntBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'HopperBulletEnv-v0', 'HopperBulletEnv-v0', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2DBulletEnv-v0']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch', 'walker-walk']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
  '''
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
  '''
  return torch.clamp((observation.div(255.0).sub(1.0)), min=-0.5+1e-20, max=0.5-1e-20)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  #return np.clip(np.floor((observation) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)
  return np.clip(observation * 255.0 + 1.0, 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  images = torch.tensor(images, dtype=torch.float32).permute((2, 0, 1))  # Resize and put channel first
  images = resize(images, [64, 64])
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  images = torch.min(torch.max(images, torch.tensor(1e-20, dtype=torch.float)), torch.tensor(1-1e-20, dtype=torch.float))
  return images.unsqueeze(dim=0)  # Add batch dimension

def _images_to_observation_dm_control(images, bit_depth):
  images = torch.tensor(np.clip(images / 255.0 - 1.0, -0.5, 0.5), dtype=torch.float32)  # Resize and put channel first
  #preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  #images = torch.min(torch.max(images, torch.tensor(1e-20, dtype=torch.float)), torch.tensor(1-1e-20, dtype=torch.float))
  return images.unsqueeze(dim=0)  # Add batch dimension

'''
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType, Waypoints

############################################################################################### 
# duplicate the functions from ultra for now until the dependency issue is resolved
import math
from scipy.spatial.distance import euclidean

from smarts.core.sensors import Observation


_WAYPOINTS = 20


# This adapter requires SMARTS to pass the next _WAYPOINTS waypoints in the agent's
# observation.
required_interface = {"waypoints": Waypoints(lookahead=_WAYPOINTS)}

def get_relative_pos(waypoint, ego_pos):
    return [waypoint.pos[0] - ego_pos[0], waypoint.pos[1] - ego_pos[1]]

def rotate2d_vector(vectors, angle):
    ae_cos = np.cos(angle)
    ae_sin = np.sin(angle)
    rot_matrix = np.array([[ae_cos, -ae_sin], [ae_sin, ae_cos]])

    vectors_rotated = np.inner(vectors, rot_matrix)
    return vectors_rotated

def get_closest_waypoint(ego_position, ego_heading, num_lookahead, goal_path):
    closest_wp = min(goal_path, key=lambda wp: wp.dist_to(ego_position))
    min_dist = float("inf")
    min_dist_idx = -1
    for i, wp in enumerate(goal_path):

        if wp.dist_to(ego_position) < min_dist:
            min_dist = wp.dist_to(ego_position)
            min_dist_idx = i
            closest_wp = wp

    waypoints_lookahead = [
        get_relative_pos(wp, ego_position)
        for wp in goal_path[
            min_dist_idx : min(min_dist_idx + num_lookahead, len(goal_path))
        ]
    ]
    if len(waypoints_lookahead) > 0:
        while len(waypoints_lookahead) < num_lookahead:
            waypoints_lookahead.append(waypoints_lookahead[-1])
    else:
        waypoints_lookahead = [
            get_relative_pos(closest_wp.pos, ego_position) for i in range(num_lookahead)
        ]

    waypoints_lookahead = rotate2d_vector(waypoints_lookahead, -ego_heading)
    return closest_wp, waypoints_lookahead

def get_closest_point_index(pts_arr, pts):
    distance = [euclidean(each, pts) for each in pts_arr]
    return np.argmin(distance)


def get_path_to_goal(goal, paths, start):
    start_pos = start.position
    path_start_pts = [each[0].pos for each in paths]

    best_path_ind = get_closest_point_index(path_start_pts, start_pos)
    path = paths[best_path_ind]

    middle_point = path[int(len(path) / 2)]
    goal_lane_id = middle_point.lane_id
    goal_lane_index = middle_point.lane_index

    path_pts = [each.pos for each in path]

    return path

def adapt(observation: Observation, reward: float) -> float:
    """Adapts a raw environment observation and an environment reward to a custom reward
    of type float.
    The raw observation from the environment must include the ego vehicle's state,
    events, and waypoint paths. See smarts.core.sensors for more information on the
    Observation type.
    Args:
        observation (Observation): The raw environment observation received from SMARTS.
        reward (float): The environment reward received from SMARTS.
    Returns:
        float: The adapted, custom reward which includes aspects of the ego vehicle's
            state and the ego vehicle's mission progress, in addition to the environment
            reward.
    """
    env_reward = reward
    ego_events = observation.events
    ego_observation = observation.ego_vehicle_state
    start = observation.ego_vehicle_state.mission.start
    goal = observation.ego_vehicle_state.mission.goal
    path = get_path_to_goal(goal=goal, paths=observation.waypoint_paths, start=start)

    linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
    angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

    # Distance to goal
    ego_2d_position = ego_observation.position[0:2]

    closest_wp, _ = get_closest_waypoint(
        num_lookahead=_WAYPOINTS,
        goal_path=path,
        ego_position=ego_observation.position,
        ego_heading=ego_observation.heading,
    )
    angle_error = closest_wp.relative_heading(
        ego_observation.heading
    )  # relative heading radians [-pi, pi]

    # Distance from center
    signed_dist_from_center = closest_wp.signed_lateral_error(
        observation.ego_vehicle_state.position
    )
    lane_width = closest_wp.lane_width * 0.5
    ego_dist_center = signed_dist_from_center / lane_width

    # NOTE: This requires the NeighborhoodVehicles interface.
    # # number of violations
    # (ego_num_violations, social_num_violations,) = ego_social_safety(
    #     observation,
    #     d_min_ego=1.0,
    #     t_c_ego=1.0,
    #     d_min_social=1.0,
    #     t_c_social=1.0,
    #     ignore_vehicle_behind=True,
    # )

    speed_fraction = max(0, ego_observation.speed / closest_wp.speed_limit)
    ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
    ego_speed_reward = min(
        0, (closest_wp.speed_limit - ego_observation.speed) * 0.01
    )  # m/s
    ego_collision = len(ego_events.collisions) > 0
    ego_collision_reward = -1.0 if ego_collision else 0.0
    ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
    ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
    ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
    ego_goal_reward = 0.0
    ego_time_out = 0.0
    ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
    ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
    ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
    # NOTE: This requires the NeighborhoodVehicles interface.
    # ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
    # NOTE: This requires the NeighborhoodVehicles interface.
    # social_safety_reward = -0.02 if social_num_violations > 0 else 0
    ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
    ego_linear_jerk = -0.0001 * linear_jerk
    ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
    env_reward /= 100
    # DG: Different speed reward
    ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
    ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0

    rewards = sum(
        [
            ego_goal_reward,
            ego_collision_reward,
            ego_off_road_reward,
            ego_off_route_reward,
            ego_wrong_way,
            ego_speed_reward,
            # ego_time_out,
            ego_dist_center_reward,
            ego_angle_error_reward,
            ego_reached_goal,
            ego_step_reward,
            env_reward,
            # ego_linear_jerk,
            # ego_angular_jerk,
            # ego_lat_speed,
            # ego_safety_reward,
            # social_safety_reward,
        ]
    )
    return rewards

######################################################################################

def ultra_default_reward_adapter(observation: Observation, reward: float) -> float:
    return adapt(observation, reward)
'''

class ControlSuiteEnv():
  def __init__(self, env, seed, max_episode_length, action_repeat, bit_depth, image_size=64):
    domain_name, task_name = env.split('-')
    self._env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
    )
    self._env.seed(seed)
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth
    if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain_name]:
      print('Using action repeat %d; recommended action repeat for domain is %d' % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain_name]))

  def reset(self):
    self.t = 0  # Reset internal timer
    observation = self._env.reset()
    return _images_to_observation_dm_control(observation, self.bit_depth)

  def step(self, action):
    action = action.detach().numpy()
    observation, reward, done, _ = self._env.step(action)
    self.t += self.action_repeat  # Increment internal timer
    done = done or self.t >= self.max_episode_length
    return _images_to_observation_dm_control(observation, self.bit_depth), reward, done

  def render(self):
    pass

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return self._env.observation_space.shape

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property
  def action_range(self):
    return (self._env.action_space.low, self._env.action_space.high)

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self._env.action_space.sample())



class GymEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    import logging
    import gym
    gym.logger.set_level(logging.ERROR)  # Ignore warnings from Gym logger
    self.symbolic = symbolic
    self._env = gym.make(env)
    self._env.seed(seed)
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
  
  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state, reward_k, done, _ = self._env.step(action)
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property
  def action_range(self):
    return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self._env.action_space.sample())

'''
class Hiway:
  def __init__(self, seed, max_episode_length, action_repeat, bit_depth, agent_id, agent_spec, scenario):
    import gym
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario],
        agent_specs={agent_id: agent_spec},
        sim_name=None,
        headless=True,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed
    )
    self._env = env
    self._agent_id = agent_id
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth
  
  def close(self):
    return self._env.close()
  
  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      states, rewards, dones, _ = self._env.step({self._agent_id: action})
      state = np.ascontiguousarray(states[self._agent_id].top_down_rgb.data)
      reward_k = rewards[self._agent_id]
      done = dones[self._agent_id]
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
      if done:
        break
    observation = _images_to_observation(state, self.bit_depth)
    return observation, reward, done
  
  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()[self._agent_id]
    return _images_to_observation(np.ascontiguousarray(state.top_down_rgb.data), self.bit_depth)
  
  def render(self):
    return self._env.render()
  
  @property
  def observation_space(self):
    return Box(0, 255, (3, 64, 64), dtype=np.uint8)
  
  @property
  def action_space(self):
    return Box(low=np.array([0,0,-1]), high=np.array([1, 1, 1]), dtype=np.float32)
  
  @property
  def observation_size(self):
    return self.observation_space.shape

  @property
  def action_size(self):
    return self.action_space.shape[0]

  @property
  def action_range(self):
    return (self.action_space.low, self.action_space.high)
  
  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self.action_space.sample())
'''

def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
  if env in GYM_ENVS:
    return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
  elif env in CONTROL_SUITE_ENVS:
    return ControlSuiteEnv(env, seed, max_episode_length, action_repeat, bit_depth)
  elif (env == 'loop' or env == '4lane'):
    '''
    AGENT_ID = "Agent-001"
    AGENT_SPEC = AgentSpec(
      interface=AgentInterface.from_type(
          AgentType.Full, max_episode_steps=1000
      ),
      agent_builder=Agent,
      reward_adapter = ultra_default_reward_adapter,
    )
    if env == '4lane':
      scenario = '/SMARTS/scenarios/intersections/4lane'
    else:
      scenario = '/SMARTS/scenarios/loop'
    return Hiway(seed, max_episode_length, action_repeat, bit_depth, AGENT_ID, AGENT_SPEC, scenario)
    '''
    raise ValueError(f"{env} is not supported")


# Wrapper for batching environments together
class EnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n
    self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
    self.dones = [True] * n

  # Resets every environment and returns observation
  def reset(self):
    observations = [env.reset() for env in self.envs]
    self.dones = [False] * self.n
    return torch.cat(observations)

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
    observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
    self.dones = dones
    observations, rewards, dones = torch.cat(observations), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    observations[done_mask] = 0
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]
