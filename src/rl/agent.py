
import torch
import numpy as np
import pandas as pd
from torch import optim
import torch.nn.functional as F 
import pickle
import matplotlib.pyplot as plt
import json

import seaborn as sns

import itertools
import gym
from src.util.auxiliaries import folderCreation
import os
from tqdm import tqdm
from gym.wrappers import Monitor
from numpy.random import default_rng
# from multiprocessing import Pool
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

MAX_WORKERS = os.cpu_count() 

try:
    from src.rl.model import Model, ActorNetwork, ValueNetwork, ActorNetwork2, ValueNetwork2
    from src.components.simulator import Simulator
except:
    import os, sys, inspect
    from pathlib import Path
    from src.rl.model import Model, ActorNetwork, ValueNetwork, ActorNetwork2, ValueNetwork2
    from src.components.simulator import Simulator
# The following code is will be used to visualize the environments.

def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

class A2CAgent:

    def __init__(self, agent_config, sim_config, env: Simulator):
        self.config = agent_config
        self.env = env
        make_seed(agent_config['seed'])
        self.env.set_seed(agent_config['seed'])
        self.cuda = agent_config['cuda'] and torch.cuda.is_available()
        self.gamma = agent_config['gamma']
        self.parallel = agent_config['parallel']
        self.folder = folderCreation("result")
        
        with open(os.path.join(self.folder, 'agent_config.json'), 'w') as fp:
            json.dump(agent_config, fp)
        with open(os.path.join(self.folder, 'simulation_config.json'), 'w') as fp:
            json.dump(sim_config, fp)

        dict_path = os.path.join(self.folder,"launch_dict") 
        with open(dict_path, 'wb') as f:
            pickle.dump(agent_config, f)
        self.test_repeat = 50
        self.gym_env = gym.wrappers.time_limit.TimeLimit
        if type(self.env) == self.gym_env:
            self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)
            self.action_space = self.env.action_space.n
            self.value_network = ValueNetwork(self.env.observation_space.shape[0], 16, 1)
            self.actor_network = ActorNetwork(self.env.observation_space.shape[0], 16, self.action_space)
        else:
            self.action_space = self.env.action_space
            self.value_network = ValueNetwork2(self.env.observation_space.shape[0], self.action_space, 32, 1)
            self.actor_network = ActorNetwork2(self.env.observation_space.shape[0], self.action_space, 32)
            
        # Our two networks
        if agent_config['load_model']:
            value_network_route = agent_config['model_route'] + 'final_value_network.pt'
            actor_network_route = agent_config['model_route'] + 'final_actor_network.pt'
            self.value_network.load_state_dict(torch.load(value_network_route))
            self.actor_network.load_state_dict(torch.load(actor_network_route))

        # Their optimizers
        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(), lr=agent_config['value_network']['learning_rate'])
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=agent_config['actor_network']['learning_rate'])
        val_network_path = os.path.join(self.folder, 'initial_value_network.pt')
        act_network_path = os.path.join(self.folder, 'initial_actor_network.pt')
        torch.save(self.value_network.state_dict(), val_network_path)
        torch.save(self.actor_network.state_dict(), act_network_path)

    def _returns_advantages(self, rewards, dones, values, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network
        
        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """
        
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
            
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self, epochs, batch_size):
        """Perform a training by batch

        Parameters
        ----------
        epochs : int
            Number of epochs
        batch_size : int
            The size of a batch
        """
        max_reward = -float('inf')
        if type(self.env) == self.gym_env:
            self.batch_size = 1000
        else:
            self.batch_size = batch_size * self.env.round
        self.epochs = epochs
        observation = self.env.reset()
        episode_count = 0
        actions = np.empty((self.batch_size,), dtype=int)
        dones = np.empty((self.batch_size,), dtype=bool)
        rewards, values = np.empty((2, self.batch_size), dtype=float)
        observations = np.empty((self.batch_size,) + (observation.shape), dtype=float)
        rewards_test = []
        rng = default_rng()
            
        for epoch in tqdm(range(epochs)):
            seedVector = rng.choice(9999999, size=batch_size, replace=False)
            with Pool(batch_size) as p:
                output = p.map(self.one_episode, seedVector)
            for i, o in enumerate(output):
                episode_observations, episode_values, episode_actions, episode_rewards, episode_dones, episode_last_observation = o 
                observations[self.env.round*i:self.env.round*(i+1)] = episode_observations
                values[self.env.round*i:self.env.round*(i+1)] = episode_values
                actions[self.env.round*i:self.env.round*(i+1)] = episode_actions
                rewards[self.env.round*i:self.env.round*(i+1)] = episode_rewards
                dones[self.env.round*i:self.env.round*(i+1)] = episode_dones
                observation = episode_last_observation

            # If our epiosde didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network(torch.unsqueeze(torch.tensor(observation, dtype=torch.float),dim =0)).detach().numpy()[0][0]
            
            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages)

            # Test it every 50 epochs
            if epoch % (50)== 0 or epoch == epochs - 1:
                output = []
                seedVector = rng.choice(9999999, size=batch_size, replace=False)
                
                with Pool(batch_size) as p:
                    output = p.map(self.evaluate, seedVector)
                rewards_test.append(np.array(output))
                print(f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')

                observation = self.env.reset()
                if rewards_test[-1].mean() > max_reward:
                    max_reward = rewards_test[-1].mean()
                    val_network_path = os.path.join(self.folder, 'value_network.pt')
                    act_network_path = os.path.join(self.folder, 'actor_network.pt')
                    torch.save(self.value_network.state_dict(), val_network_path)
                    torch.save(self.actor_network.state_dict(), act_network_path)

        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        eval_result_path = os.path.join(self.folder, 'evaluation_result.pkl')
        image_path = os.path.join(self.folder, 'evaluation_graph.png')
        val_network_path = os.path.join(self.folder, 'final_value_network.pt')
        act_network_path = os.path.join(self.folder, 'final_actor_network.pt')
        torch.save(self.value_network.state_dict(), val_network_path)
        torch.save(self.actor_network.state_dict(), act_network_path)

        with open(eval_result_path, 'wb') as f:
            pickle.dump(rewards_test, f)
        plt.savefig(image_path)

    def one_episode(self, seed):
        make_seed(seed)
        episode_size = self.env.round
        observation = self.env.reset()
        self.env.set_seed(seed)
        actions = np.empty((episode_size,), dtype=int)
        dones = np.empty((episode_size,), dtype=bool)
        rewards, values = np.empty((2, episode_size), dtype=float)
        observations = np.empty((episode_size,) + (observation.shape), dtype=float)
        for i in range(episode_size):
            observations[i, :]            = observation
            values[i]                     = self.value_network(torch.unsqueeze(torch.tensor(observations[i], dtype=torch.float), dim = 0)).detach().numpy()
            policy                        = self.actor_network(torch.unsqueeze(torch.tensor(observations[i], dtype=torch.float), dim = 0))
            actions[i]                    = np.squeeze(torch.multinomial(policy, 1).detach().numpy())
            observation, rewards[i], dones[i] = self.env.step(actions[i])
            if dones[i]:
                observation                = self.env.reset()
            last_observation               = observation
        return observations, values, actions, rewards, dones, last_observation

    def optimize_model(self, observations, actions, returns, advantages):

        actions = F.one_hot(torch.tensor(actions), self.action_space)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)

        if self.cuda:
            self.value_network.cuda()
            self.actor_network.cuda()
            actions  = actions.cuda()
            returns = returns.cuda()
            advantages = advantages.cuda()
            observations = observations.cuda()

        # MSE for the values
        self.value_network_optimizer.zero_grad()
        values = self.value_network(observations)
        loss_value = 1 * F.mse_loss(values, returns)
        loss_value.backward()
        self.value_network_optimizer.step()

        # Actor loss
        self.actor_network_optimizer.zero_grad()
        policies = self.actor_network(observations)
        loss_policy = ((actions.float() * policies.log()).sum(-1) * advantages).mean()
        loss_entropy = - (policies * policies.log()).sum(-1).mean()
        loss_actor = - loss_policy - 0.0001 * loss_entropy
        loss_actor.backward()
        self.actor_network_optimizer.step()
        if self.cuda:
            self.value_network.cpu()
            self.actor_network.cpu()
        return loss_value, loss_actor    

    def evaluate(self, seed):
        make_seed(seed)
        observation = self.env.reset()
        reward_episode = 0
        done = False
        i = 0
        
        while not done:
            observations = observation
            policy = self.actor_network(torch.unsqueeze(torch.tensor(observations, dtype=torch.float), dim = 0))
            action = np.squeeze(torch.multinomial(policy, 1).detach().numpy())
            observation, reward, done = self.env.step(action)
            reward_episode += reward
            i += 1
            
        return reward_episode