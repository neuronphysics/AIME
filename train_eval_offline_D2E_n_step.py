"""Training and evaluation in the offline mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.tensorboard import SummaryWriter
import re
import datetime
import utils_planner as utils
import gin.torch.external_configurables
from typing import Dict, Any
from DataCollectionD2E_n_step import *
from planner_D2E_regularizer_n_step import eval_policies, D2EAgent, Config

config: Dict[str, Any] = {}


# gin.clear_config()
def get_datetime():
    now = datetime.datetime.now().isoformat()
    now = re.sub(r'\D', '', now)[:-6]
    return now


class CustomUnpickler(dill.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-777059657
    def __init__(self, *args, map_location="cpu", **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


@gin.configurable
def train_eval_offline(
        # Basic args.
        log_dir,
        data_file,
        env_name='HalfCheetah-v2',
        n_train=int(1e6),
        shuffle_steps=0,
        seed=0,
        use_seed_for_data=False,
        # Train and eval args.
        total_train_steps=int(1e6),
        summary_freq=100,
        print_freq=1000,
        plot_train_freq=1000,
        save_freq=int(2e4),
        # todo
        eval_freq=10,
        n_eval_episodes=20,
        # Agent args.
        model_params=(((200, 200),), 2, 1),
        optimizers=((0.0001, 0.5, 0.99),),
        batch_size=256,
        weight_decays=(0.0,),
        update_freq=1,
        update_rate=0.005,
        discount=0.99,
        done=False,
        device=None
):
    ###Training a policy with a fixed dataset.###
    # Create tf_env to get specs.
    dm_env = gym.spec(env_name).make()
    env = alf_gym_wrapper.AlfGymWrapper(dm_env, discount=discount)
    env = TimeLimit(env, MUJOCO_ENVS_LENNGTH[env_name])
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare data.
    logging.info('Loading data from %s ...', data_file)

    data_ckpt_name = os.path.join(data_file, 'data_{}.pt'.format(env_name))
    whole_data_ckpt_name = os.path.join(data_file, 'data_{}.pth'.format(env_name))

    data_size, state = torch.load(data_ckpt_name, map_location=device)

    if os.path.getsize(whole_data_ckpt_name) > 0:
        with open(whole_data_ckpt_name, "rb") as f:
            # if file is not empty scores will be equal
            # to the value unpickled
            full_data = CustomUnpickler(f, map_location=device).load()

    logging.info('Loading data from dataset with size %d , %d ...', data_size, full_data.size)
    for k, v in full_data._config.items():
        if k == 'observation_spec':
            full_data._config['observation_spec'] = observation_spec
        elif k == 'action_spec':
            full_data._config['action_spec'] = action_spec

    # Split data.
    n_train = min(n_train, full_data.size)
    logging.info('n_train %s.', n_train)
    if use_seed_for_data:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState(0)

    shuffled_indices = shuffle_indices_with_steps(
        n=full_data.size, steps=shuffle_steps, rand=rand)

    train_indices = shuffled_indices[:n_train]
    train_data = full_data.create_view(train_indices)
    logging.info('DEBUG: creating agent ....')
    # Create agent.
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
        env_name=env_name,
        train_data=train_data)

    agent_args = Config(agent_flags).agent_args
    logging.info('DEBUG: Initialize a brac agent ....')
    agent = D2EAgent(**vars(agent_args))  # ATTENTION: Debugg
    agent_ckpt_name = os.path.join(log_dir, 'agent')

    # Restore agent from checkpoint if there exists one.
    if os.path.exists('{}.index'.format(agent_ckpt_name)):
        logging.info('Checkpoint found at %s.', agent_ckpt_name)
        torch.load(agent, agent_ckpt_name)

    # Train agent.
    train_summary_dir = os.path.join(log_dir, 'train')
    eval_summary_dir = os.path.join(log_dir, 'eval')
    train_summary_writer = SummaryWriter(log_dir=train_summary_dir)
    eval_summary_writers = collections.OrderedDict()
    for policy_key in agent.test_policies.keys():
        eval_summary_writer = SummaryWriter(
            log_dir=os.path.join(eval_summary_dir, policy_key))
        eval_summary_writers[policy_key] = eval_summary_writer
    eval_results = []

    time_st_total = time.time()
    time_st = time.time()
    step = agent.global_step
    timed_at_step = step
    while step < total_train_steps:
        print(f"at step {int(step)}/{total_train_steps}")
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
                env, agent.test_policies, n_eval_episodes)
            eval_results.append([step] + eval_result)
            logging.info('Testing at step %d:', step)
            for policy_key, policy_info in eval_infos.items():
                logging.info(utils.get_summary_str(
                    step=None, info=policy_info, prefix=policy_key + ': '))
                utils.write_summary(eval_summary_writers[policy_key], policy_info, step)
            time_st = time.time()
            timed_at_step = step
        if step % save_freq == 0:
            agent.checkpoint_path = agent_ckpt_name
            agent._build_checkpointer()
            logging.info('Agent saved at %s.', agent_ckpt_name)

    agent._build_checkpointer()
    time_cost = time.time() - time_st_total
    logging.info('Training finished, time cost %.4gs.', time_cost)
    logging.info("{}......".format(eval_results[0]))
    return torch.cat([x.unsqueeze(0) for x in eval_results[0][1:]], dim=-1)


def main(args):
    logging.set_verbosity(logging.INFO)

    # Setup data file path.
    data_dir = os.path.join(
        args.data_root_offlinerl_dir,
        args.env_name,
        args.data_name,
        args.data_sub_offlinerl_dir,
    )
    data_file = os.path.join(
        data_dir, args.data_file_name)
    logging.info('Data directory %s.', args.data_root_offlinerl_dir)
    # Setup log dir.
    if args.sub_dir == 'auto':
        sub_dir = get_datetime()
    else:
        sub_dir = args.sub_dir
    log_dir = os.path.join(
        args.data_root_offlinerl_dir,
        args.env_name,
        args.data_name,
        'n' + str(args.n_train),
        args.agent_name,
        sub_dir,
        str(args.seed),
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        pass

    train_eval_offline(
        log_dir=log_dir,
        data_file=data_file,
        env_name=args.env_name,
        n_train=args.n_train,
        total_train_steps=args.total_train_steps,
        n_eval_episodes=args.n_eval_episodes,
        print_freq=args.print_freq,
        summary_freq=args.summary_freq,
        plot_train_freq=args.plot_train_freq,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )


def Train_offline_D2E(args):
    data_dir = 'offlinerl'
    # args.test_srcdir = os.getcwd()
    args.data_root_offlinerl_dir = os.path.join(args.test_srcdir, data_dir)
    args.data_sub_offlinerl_dir = '0'
    # args.env_name = 'Pendulum-v0'
    args.data_name = 'example'
    args.agent_name = 'DreamToExplore'
    # size of neural networks for actor critic and value networks, number of Q networks and number of value nnetwork
    args.gin_bindings = [
        'train_eval_offline.model_params=((200, 200), 2, 1)',
        'train_eval_offline.optimizers=((5e-4, 0.5, 0.99),)']
    args.n_train = 20000
    args.n_eval_episodes = 50
    args.total_train_steps = 100000  # Short training.

    main(args)  # Just test that it runs.


if __name__ == "__main__":

    repo_dir = "/mnt/e/pycharm_projects/AIME"
    # repo_dir = "/home/p23shi/D2E/AIME-comb/AIME"
    ##############################
    if not os.path.exists(os.path.join(os.getenv('HOME', '/'), repo_dir, 'offlinerl/learn')):
        os.makedirs(os.path.join(os.getenv('HOME', '/'), repo_dir, 'offlinerl/learn'))
    else:
        pass
    ###train_offline.py
    parser = argparse.ArgumentParser(description='DreamToExplore',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root_offlinerl_dir', type=dir_path,
                        default=os.path.join(os.getenv('HOME', '/'), repo_dir, 'offlinerl'),
                        help='Root directory for data.')
    parser.add_argument('--data_sub_offlinerl_dir', type=str, default=None, help='')
    parser.add_argument('--test_srcdir', type=str, default=str(os.path.join(os.getenv('HOME', '/'), repo_dir)),
                        help='directory for saving test data.')
    parser.add_argument('--data_name', type=str, default='eps1', help='data name.')
    parser.add_argument('--data_file_name', type=str, default='', help='data checkpoint file name.')

    # Flags for offline training.
    parser.add_argument('--root_dir', type=dir_path,
                        default=os.path.join(os.getenv('HOME', '/'), repo_dir, 'offlinerl/learn'),
                        help='Root directory for writing logs/summaries/checkpoints.')
    parser.add_argument('--sub_dir', type=str, default='0', help='')
    parser.add_argument('--print_freq', type=int, default=100, help='print train info frequency')
    parser.add_argument('--summary_freq', type=int, default=1000, help='write summary frequency')
    parser.add_argument('--plot_train_freq', type=int, default=100, help='plot train info frequency')
    parser.add_argument('--agent_name', type=str, default='DreamToExplore', help='agent name.')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='env name.')
    parser.add_argument('--seed', type=int, default=0, help='random seed, mainly for training samples.')
    parser.add_argument('--total_train_steps', type=int, default=int(5e5), help='')
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='')
    parser.add_argument('--n_train', type=int, default=int(1e6), help='')
    parser.add_argument("--gin_file", type=str, default=[], nargs='*', help='Paths to the gin-config files.')
    parser.add_argument('--gin_bindings', type=str, default=[], nargs='*', help='Gin binding parameters.')
    args = parser.parse_args()
    # print(args)
    config.update(vars(parser.parse_args()))
    logging.info("Parsed %i arguments.", len(config))
    # args = parser.parse_args(sys.argv[1:])
    args = parser.parse_args()
    # print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    # gin.parse_config_files_and_bindings(args.gin_file, args.gin_bindings)
    Train_offline_D2E(args)
    # gin.clear_config()
    # gin.config._REGISTRY.clear()
