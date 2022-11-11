import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from preprocessing import make_environment


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, gamma = 0.99, tau = 1.0):
        super(ActorCriticNetwork, self).__init__()
        self.gamma = gamma
        self.tau = tau

<<<<<<< HEAD
        '''
        Potentially need to break up the shared input layers
        to form two seperate actor and critic networks in this class.
        '''

=======
>>>>>>> master
        # Convolutional Network:
        self.cv1 = nn.Conv2d(input_dims[0], 32, 3, stride = 2, padding = 1)
        self.cv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.cv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.cv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)

        output_conv_shape = self.calc_conv_output_dims(input_dims)

<<<<<<< HEAD
        # print(f"\n\noutput_conv_shape: {output_conv_shape}\n\n")

=======
>>>>>>> master
        self.gru = nn.GRUCell(output_conv_shape, 256)

        # Actor - Policy:
        self.pi = nn.Linear(256, n_actions)

        # Critic - Value:
        self.V = nn.Linear(256, 1)

    def calc_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.cv1(state)
        dims = self.cv2(dims)
        dims = self.cv3(dims)
        dims = self.cv4(dims)
        return int(np.prod(dims.size()))

    def forward(self, state, hidden_state):
<<<<<<< HEAD
        # print(f"\n\n\n\nIN FORWARD")
        # print(f"len(state): {len(state)}")
        # print(f"len(hidden_state): {len(hidden_state)}")

=======
>>>>>>> master
        conv1 = F.elu(self.cv1(state))
        conv2 = F.elu(self.cv2(conv1))
        conv3 = F.elu(self.cv3(conv2))
        conv4 = F.elu(self.cv3(conv3))
        # conv4 shape is BS * n_filters * H * W
        # must reshape conv3 to BS * num_input_features to pass to fc1
		# conv4 shape is BS x n_filters x H x W

        conv_state = conv4.view((conv4.size()[0], -1))
<<<<<<< HEAD
        # conv_state = conv4.view(conv4.size()[0], -1)
        # conv_state = conv4.view(conv4.size()[0], -1) # conv4.size()[0] = BS, -1 = flatten all other dimensions CURRENT
        # flat_state = T.flatten(conv4) # (CURRENT) USING THIS NOW

        # print(f"\n\n\ntype(conv_state): {type(conv_state)}")
        # print(f"conv_state.size(): {conv_state.size()}\n\n\n")

        # hidden_state = self.gru(conv_state, (hidden_state))
        # hidden_state = T.flatten(hidden_state) # (CURRENT)

        # hidden_state = self.gru(flat_state, (hidden_state)) # (CURRENT)
=======
>>>>>>> master
        hidden_state = self.gru(conv_state, (hidden_state))

        pi = self.pi(hidden_state)
        V = self.V(hidden_state)

        # action selection:
        action_probs = T.softmax(pi, dim = 1)
        # action_probs = T.softmax(pi, dim = 0) # (CURRENT)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

<<<<<<< HEAD
        # print(f"\n\n\n\naction: {action}, V: {V}, log_prob: {log_prob}, hidden_state: {hidden_state}\n\n\n\n")

=======
>>>>>>> master
        return action.numpy()[0], V, log_prob, hidden_state
        # return action, V, log_prob, hidden_state # (CURRENT)

    def calc_returns(self, done, rewards, values):
        # handles batch states or single states
        values = T.cat(values).squeeze()

<<<<<<< HEAD
        if len(values.size()) == 1: # we have a batch of states
=======
        if len(values.size()) == 1: # batch of states
>>>>>>> master
            R = values[-1] * (1 - int(done))
        elif len(values.size()) == 0: # single state
            R = values * (1 - int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype = T.float).reshape(
            values.size()
        )

        return batch_return

    def calc_cost(self, new_state, hidden_state, done, rewards, values, log_probs):
        returns = self.calc_returns(done, rewards, values)

        next_V = T.zeros(1, 1) if done else self.forward(
            T.tensor(np.array([new_state]), dtype = T.float), hidden_state
        )[1]

        '''
        next_V = T.zeros(1, 1) if done else self.forward(
            T.tensor([new_state], dtype = T.float), hidden_state
        )[1]
        '''

        values.append(next_V.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)

<<<<<<< HEAD
        # print(f"\n\nlog_probs.grad_fn: {log_probs.grad_fn}\n\n")

=======
>>>>>>> master
        rewards = T.tensor(rewards)

        delta_t = rewards + self.gamma + values[1:] - values[:-1]
        n_steps = len(delta_t)
        generalized_advantage_estimate = np.zeros(n_steps)

        for t in range(n_steps):
            for k in range(0, n_steps - t):
                temp = (self.gamma * self.tau) ** k * delta_t[t + k]
                generalized_advantage_estimate[t] += temp

        generalized_advantage_estimate = T.tensor(
            generalized_advantage_estimate, dtype = T.float
        )

        actor_loss = -(log_probs * generalized_advantage_estimate).sum()
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = (-log_probs * T.exp(log_probs)).sum()
        total_loss = actor_loss + critic_loss - 0.01 * entropy_loss

<<<<<<< HEAD
        # This whole line has been added - almost certainly a better way to do this though
        # total_loss = T.tensor(total_loss, dtype = T.float, requires_grad = True) ### --------- HERE --------- ###

        return total_loss








    # def calc_cost(self, new_state, hidden_state, done, rewards, values, log_probs):
    #     returns = self.calc_returns(done, rewards, values)
    #
    #     next_V = T.zeros(1, 1) if done else self.forward(
    #         T.tensor([new_state], dtype = T.float), hidden_state
    #     )[1]
    #
    #     values.append(next_V.detach())
    #     values = T.cat(values).squeeze()
    #     log_probs = T.cat(log_probs)
    #
    #     print(f"\n\nlog_probs.grad_fn: {log_probs.grad_fn}\n\n")
    #
    #     # print(f"\n\n\nlog_probs.requires_grad: {log_probs.requires_grad}")
    #     # print(f"log_probs.is_leaf: {log_probs.is_leaf}")
    #     # print(f"log_probs grad_fn: {log_probs.grad_fn}\n\n\n")
    #
    #     rewards = T.tensor(rewards)
    #
    #     delta_t = rewards + self.gamma + values[1:] - values[:-1]
    #     # values[1:] (at time t + 1), values[:-1] (at time t)
    #     n_steps = len(delta_t)
    #     generalized_advantage_estimate = np.zeros(n_steps)
    #
    #     for t in range(n_steps):
    #         for k in range(0, n_steps - t):
    #             temp = (self.gamma * self.tau) ** k * delta_t[t + k]
    #             generalized_advantage_estimate[t] += temp
    #
    #     generalized_advantage_estimate = T.tensor(
    #         generalized_advantage_estimate, dtype = T.float
    #     )
    #
    #
    #     # actor_loss = -(log_probs * generalized_advantage_estimate) (CURRENT)
    #     actor_loss = -(log_probs * generalized_advantage_estimate).sum()
    #     # print(f"\n\nactor_loss requires grad: {actor_loss.requires_grad}")
    #     # print(f"actor_loss is_leaf: {actor_loss.is_leaf}")
    #     # print(f"actor_loss grad_fn: {actor_loss.grad_fn}")
    #
    #     # clculate the critic loss:
    #     critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
    #     # print(f"critic_loss requires grad: {critic_loss.requires_grad}")
    #     # print(f"critic_loss is_leaf: {critic_loss.is_leaf}")
    #     # print(f"critic_loss grad_fn: {critic_loss.grad_fn}")
    #
    #     # calculate the total loss (as per the paper):
    #     entropy_loss = (-log_probs * T.exp(log_probs)).sum()
    #     # print(f"entropy_loss requires grad: {entropy_loss.requires_grad}")
    #     # print(f"entropy_loss is_leaf: {entropy_loss.is_leaf}")
    #     # print(f"entropy_loss grad_fn: {entropy_loss.grad_fn}")
    #
    #     total_loss = actor_loss + critic_loss - 0.01 * entropy_loss # "requires_grad" is true not a leaf
    #     # print(f"\n\ntotal_loss requires grad: {total_loss.requires_grad}")
    #     # print(f"total_loss is_leaf: {total_loss.is_leaf}")
    #     # print(f"total_loss grad_fn: {total_loss.grad_fn}\n\n")
    #
    #     # This whole line has been added - almost certainly a better way to do this though
    #     # total_loss = T.tensor(total_loss, dtype = T.float, requires_grad = True) ### --------- HERE --------- ###
    #
    #     return total_loss
=======
        return total_loss
>>>>>>> master
