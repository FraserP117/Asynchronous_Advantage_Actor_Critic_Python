
import numpy as np
import torch as T
from A3C_Agent import ActorCriticNetwork
from preprocessing import make_environment
from memory import Memory
from utils import plot_learning_curve
<<<<<<< HEAD
=======
# from gym.utils.step_api_compatibility import step_api_compatibility
>>>>>>> master

'''
set multiprocessing variables
set input_shape, n_threads, n_actions (Pong has 6)
use mp.Value('i', 0) for shared episode or time step (initial value 0)
call ParallelEnv constructor

ParallelEnv modifications:
    * pass n_actions, input_shape and global_index
    * global agent and optimizer; share_memory()
    * global agent params: lr = 1e-4
    * pass global agent, optimizer and global_index to worker
    * worker function instantiates a local agent

Worker Modifications:
    * local env needs correct frame buffer
    * main loop calculates losses/gradients and applies
    * T.nn.utils.clip_grad_norm(parameters, 40)
    * detatch hidden_state before backprop
    * T_max = 20
    * total episodes = 1000
'''

def worker(name, input_shape, n_actions, global_agent, optimizer, env_id,
    n_threads, global_idx):
    T_max = 20
    local_agent = ActorCriticNetwork(input_shape, n_actions)
    memory = Memory()

    # input shape has # channels first but the frame buffer must have # channels last
    # swap the channels:
    frame_buffer = [input_shape[1], input_shape[2], 1]
    env = make_environment(env_id, shape = frame_buffer)

    episode, max_eps, t_steps, scores = 0, 1000, 0, []

    while episode < max_eps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0
        hidden_state = T.zeros(1, 256)
        while not done:

<<<<<<< HEAD
            obs = T.tensor(np.array([obs]), dtype = T.float)
            # obs = T.tensor([obs], dtype = T.float) # OG Version

            action, value, log_prob, hidden_state = local_agent(obs, hidden_state)
            next_obs, reward, done, info = env.step(action)
=======
            # obs = T.tensor(np.array([obs]), dtype = T.float)
            obs = T.tensor([obs], dtype = T.float)

            action, value, log_prob, hidden_state = local_agent(obs, hidden_state)

            # new gym step API:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            '''
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached....
            '''
>>>>>>> master

            memory.store_transition(reward, value, log_prob)
            score += reward
            ep_steps += 1
            t_steps += 1
            obs = next_obs

            if ep_steps % T_max == 0 or done:
                rewards, values, log_probs = memory.sample_memory() # log_probs is a python lists of pytorch tensors

                loss = local_agent.calc_cost(
                    obs, hidden_state, done, rewards, values, log_probs
                )

                optimizer.zero_grad()
                hidden_state = hidden_state.detach_()

<<<<<<< HEAD
                ###
                # loss.retain_grad() ### Added line
                ###

                loss.backward() ### ------------------------ THE ISSUE WITH THE GRADIENTS IS HERE ------------------------ ###

                # loss.sum().backward() # CURRENT
=======

                loss.backward() ### ------------------------ THE ISSUE WITH THE GRADIENTS IS HERE ------------------------ ###

                # loss.sum().backward()
>>>>>>> master
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                                                local_agent.parameters(),
                                                global_agent.parameters()):
                    global_param._grad = local_param.grad

                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())
                memory.clear_memory()

        episode += 1

        with global_idx.get_lock():
            global_idx.value += 1

        if name == '1':
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print(f'A3C episode: {episode}, thread: {name} of {n_threads}, steps: {t_steps/1e6}, score: {score}, avg_score: {avg_score}')

    if name == '1':
        x = [z for z in range(episode)]
        plot_learning_curve(x, scores, 'A3C_pong_final.png')
