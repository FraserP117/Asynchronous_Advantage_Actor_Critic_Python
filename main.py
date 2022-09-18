import os
import torch.multiprocessing as mp
from parallel_environment import ParallelEnv
from utils import plot_learning_curve

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    global_episodes = mp.Value('i', 0)
    env_id = 'PongNoFrameskip-v4'
    # env_id = 'CartPole-v0'
    n_threads = 4
    n_actions = 6
    input_shape = [4, 42, 42]
    env = ParallelEnv(
        env_id = env_id, num_threads = n_threads, n_actions = n_actions,
        global_idx = global_episodes, input_shape = input_shape
    )
