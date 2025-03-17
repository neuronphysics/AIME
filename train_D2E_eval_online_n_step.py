"""Training and evaluation in the online mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
import utils_planner as utils
from torch.utils.tensorboard import SummaryWriter
import datetime
import re
from typing import Dict, Any
from DataCollectionD2E_n_step import *
from planner_D2E_regularizer_n_step import D2EAgent, Config, eval_policies, ContinuousRandomPolicy
import sys
from pathlib import Path
config: Dict[str, Any] = {}


# gin.clear_config()
def get_datetime():
    now = datetime.datetime.now().isoformat()
    now = re.sub(r'\D', '', now)[:-6]
    return now


@gin.configurable
def train_eval_online(
        # Basic args.
        log_dir,
        env_name='HalfCheetah-v2',
        # Train and eval args.
        total_train_steps=int(3e6),
        summary_freq=100,
        print_freq=1000,
        plot_train_freq=500,
        save_freq=int(1e8),
        eval_freq=5000,
        n_eval_episodes=20,
        # For saving a partially trained policy.
        eval_target=None,  # Target return value to stop training.
        eval_target_n=2,  # Stop after n consecutive evals above eval_target.
        # Agent train args.
        initial_explore_steps=20000,
        replay_buffer_size=int(1e6),
        model_params=(((200, 200),), 2, 1),
        optimizers=((0.0001, 0.9, 0.999),),
        batch_size=128,
        weight_decays=(0.0,),
        update_freq=1,
        update_rate=0.005,
        discount=0.99,
        done=False,
        device=None,
        group_size=15
):
    """Training a policy with online interaction."""
    # Create env to get specs.
    dm_env = gym.spec(env_name).make()
    env = alf_gym_wrapper.AlfGymWrapper(dm_env, discount=discount)
    env = TimeLimit(env, MUJOCO_ENVS_LENNGTH[env_name])
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    env_test = alf_gym_wrapper.AlfGymWrapper(dm_env, discount=discount)
    env_test = TimeLimit(env_test, MUJOCO_ENVS_LENNGTH[env_name])
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize wm_image_replay_buffer.
    train_data = Dataset(
        observation_spec,
        action_spec,
        replay_buffer_size,
        circular=True,
        group_size=group_size
    )

    data_ckpt_name = os.path.join(log_dir, 'replay_{}'.format(env_name))

    time_st_total = time.time()
    time_st = time.time()
    timed_at_step = 0

    # Collect data from random policy.
    explore_policy = ContinuousRandomPolicy(action_spec)
    steps_collected = 0
    log_freq = 5000
    logging.info('Collecting data ...')
    time_step = env.reset()
    collector = DataCollector(env, explore_policy, train_data)
    while steps_collected < initial_explore_steps:
        count = collector.collect_transition(group_size)
        steps_collected += count
        if (steps_collected % log_freq == 0
            or steps_collected == initial_explore_steps) and count > 0:
            steps_per_sec = ((steps_collected - timed_at_step)
                             / (time.time() - time_st))
            timed_at_step = steps_collected
            time_st = time.time()
            logging.info('(%d/%d) steps collected at %.4g steps/s.', steps_collected,
                         initial_explore_steps, steps_per_sec)

    # Construct agent.
    agent_flags = utils.Flags(
        observation_spec=observation_spec,
        action_spec=action_spec,
        model_params=model_params,
        optimizers=optimizers,
        batch_size=batch_size,
        weight_decays=weight_decays,
        update_freq=update_freq,
        update_rate=update_rate,
        discount=discount,
        done=done,
        env_name=env,
        train_data=train_data)
    agent_args = Config(agent_flags).agent_args
    agent = D2EAgent(**vars(agent_args))

    # Prepare savers for models and results.
    train_summary_dir = os.path.join(log_dir, 'train')
    eval_summary_dir = os.path.join(log_dir, 'eval')
    train_summary_writer = SummaryWriter(
        train_summary_dir)
    eval_summary_writers = collections.OrderedDict()
    for policy_key in agent.test_policies.keys():
        eval_summary_writer = SummaryWriter(
            os.path.join(eval_summary_dir, policy_key))
        eval_summary_writers[policy_key] = eval_summary_writer
    agent_ckpt_name = os.path.join(log_dir, 'agent')
    eval_results = []

    # Train agent.
    logging.info('Start training ....')
    time_st = time.time()
    timed_at_step = 0
    target_partial_policy_saved = False
    collector = DataCollector(
        env, agent.online_policy, train_data)
    for _ in range(total_train_steps):

        collector.collect_transition(group_size)
        agent.train_step()
        step = agent.global_step
        if step % summary_freq == 0 or step == total_train_steps:
            agent.write_train_summary(train_summary_writer)
        if step % print_freq == 0 or step == total_train_steps:
            agent.print_train_info()
        if step % plot_train_freq == 0 or step == total_train_steps:
            agent.plot_train_info(os.path.join(log_dir, "train"))

        if step % eval_freq == 0 or step == total_train_steps:
            time_ed = time.time()
            time_cost = time_ed - time_st
            logging.info(
                'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
            eval_result, eval_infos = eval_policies(
                env_test, agent.test_policies, n_eval_episodes)
            eval_results.append([step] + eval_result)
            # Decide whether to save a partially trained policy based on current model
            # performance.
            if (eval_target is not None and len(eval_results) >= eval_target_n
                    and not target_partial_policy_saved):
                evals_ = list([eval_results[-(i + 1)][1]
                               for i in range(eval_target_n)])
                evals_ = np.array(evals_)
                if np.min(evals_) >= eval_target:
                    agent.save(agent_ckpt_name + '_partial_target')
                    save_copy(train_data, data_ckpt_name + '_partial_target.pth')
                    logging.info('A partially trained policy was saved at step %d,'
                                 ' with episodic return %.4g.', step, evals_[-1])
                    target_partial_policy_saved = True
            logging.info('Testing at step %d:', step)
            ##
            for policy_key, policy_info in eval_infos.items():
                logging.info(utils.get_summary_str(
                    step=None, info=policy_info, prefix=policy_key + ': '))
                utils.write_summary(eval_summary_writers[policy_key], policy_info, step)
            time_st = time.time()
            timed_at_step = step
        if step % save_freq == 0:
            agent.save(agent_ckpt_name + '-' + str(step))

    # Final save after training.
    agent.save(agent_ckpt_name + '_final')
    torch.save(train_data.state_dict(), data_ckpt_name + '_final.pth')
    time_cost = time.time() - time_st_total
    logging.info('Training finished, time cost %.4gs.', time_cost)
    return np.array(eval_results)


def main(args):
    logging.set_verbosity(logging.INFO)
    # gin.parse_config_files_and_bindings(args.gin_file, args.gin_bindings)
    if args.sub_dir == 'auto':
        sub_dir = get_datetime()
    else:
        sub_dir = args.sub_dir
    log_dir = os.path.join(
        args.root_dir,
        args.env_name,
        args.agent_name,
        sub_dir,
    )
    # utils.maybe_makedirs(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        pass

    train_eval_online(
        log_dir=log_dir,
        env_name=args.env_name,
        total_train_steps=args.total_train_steps,
        n_eval_episodes=args.n_eval_episodes,
        eval_target=args.eval_target,
        group_size=args.group_size
    )


if __name__ == '__main__':
    repo_dir = Path.cwd()

    if not os.path.exists(os.path.join(os.getenv('HOME', '/'), repo_dir, "online")):
        os.makedirs(os.path.join(os.getenv('HOME', '/'), repo_dir, "online"))

    parser = argparse.ArgumentParser(description='DreamToExplore',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root_dir', type=dir_path, default=os.path.join(os.getenv('HOME', '/'),
                                                                          repo_dir, "online"),
                        help='Root directory for writing logs/summaries/checkpoints.')
    parser.add_argument('--sub_dir', type=str, default='0', help='')

    parser.add_argument('--agent_name', type=str, default='sac', help='agent name.')
    parser.add_argument('--eval_target', type=int, default=1000, help='threshold for a paritally trained policy')

    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='env name.')
    parser.add_argument('--seed', type=int, default=0, help='random seed, mainly for training samples.')
    parser.add_argument('--total_train_steps', type=int, default=int(5e6), help='')
    parser.add_argument('--n_eval_episodes', type=int, default=int(20), help='')

    parser.add_argument('--gin_file', type=str, default=[], nargs='*', help='Paths to the gin-config files.')
    parser.add_argument('--gin_bindings', type=str, default=[], nargs='*', help='Gin binding parameters.')
    parser.add_argument('--group_size', type=int, default=50, help='number of steps required for general advantage'
                                                                   ' estimator')
    args = parser.parse_args(sys.argv[1:])
    config.update(vars(args))
    logging.info("Parsed %i arguments.", len(config))
    main(args)
