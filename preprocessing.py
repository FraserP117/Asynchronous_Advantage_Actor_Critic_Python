from collections import deque
import cv2
import numpy as np
import gym

'''
Transform 3 channels to 1 - gray scale
Downscale to 42*42 from 84*84
repeat action 4 times
swap channels to first position
stack 4 most recent frames
scale inputs by 255
'''

class RepeatAction(gym.Wrapper):
    def __init__(self, env = None, repeat = 4, fire_first = False):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat
        self.fire_first = fire_first
        self.shape = env.observation_space.low.shape
        # self.env = env

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):

            # new gym step API:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated

            '''
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached....
            '''

            total_reward += reward
            if done:
                break

        return next_obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'

            # new gym step API:
            obs, _, _, _, _ = self.env_step(1)

        return obs


class FramePreProcessor(gym.ObservationWrapper):
    def __init__(self, shape, env = None):
        super(FramePreProcessor, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1]) # set shape by swapping channel axis
        self.observation_space = gym.spaces.Box(
            low = 0.0, high = 1.0, shape = self.shape, dtype = np.float32
        ) # set obs space to new shape using gym.spaces.Box(0 to 1.0)

    def observation(self, raw_obs):
        new_frame = cv2.cvtColor(raw_obs, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(new_frame, self.shape[1:], interpolation = cv2.INTER_AREA)
        new_obs = np.array(resized_frame, dtype = np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs


class FrameStacker(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(FrameStacker, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis = 0),
            env.observation_space.high.repeat(repeat, axis = 0),
            dtype = np.float32
        )
        # self.env = env
        self.frame_stack = deque(maxlen = repeat)

    def reset(self):
        self.frame_stack.clear()

        obs = self.env.reset()

        obs = obs[0] # just grabbing the pixel values in the image to stack.

        for i in range(self.frame_stack.maxlen):
            self.frame_stack.append(obs)

        # np_frame_stack = np.array(self.frame_stack)
        np_frame_stack = np.array(self.frame_stack, dtype = np.float32)

        return np_frame_stack.reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.frame_stack.append(observation)
        # self.frame_stack = np.append(self.frame_stack, observation)
        return np.array(self.frame_stack).reshape(self.observation_space.low.shape)


def make_environment(env_name, shape = (42, 42, 1), repeat = 4):
    env = gym.make(env_name)
    env = RepeatAction(env, repeat)
    env = FramePreProcessor(shape, env)
    env = FrameStacker(env, repeat)

    return env
