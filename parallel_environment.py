
import torch.multiprocessing as mp
from A3C_Agent import ActorCriticNetwork
from shared_adam_optimizer import SharedAdamOptimizer
from worker import worker

class ParallelEnv():
    def __init__(self, env_id, global_idx, input_shape, n_actions, num_threads):
        names = [str(i) for i in range(num_threads)]

        global_actor_critic = ActorCriticNetwork(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdamOptimizer(global_actor_critic.parameters(), lr = 1e-4)
        self.processes = [
            mp.Process(
                target = worker,
                args = (
                    name, input_shape, n_actions, global_actor_critic,
                    global_optim, env_id, num_threads, global_idx
                )
            ) for name in names
        ]
        [p.start() for p in self.processes]
        [p.join() for p in self.processes]
