
import torch
import numpy as np
import pandas as pd
from torch import optim
import torch.nn.functional as F 

import seaborn as sns

import itertools
import gym
import os
from tqdm import tqdm
from gym.wrappers import Monitor
from numpy.random import default_rng
from multiprocessing import Pool
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

    def __init__(self, config, sim_config):
        self.config = config
        self.env = gym.make(config['env_id'])
        self.env2 = Simulator(**sim_config)
        belief = self.env2.reset()
        input_dim = belief.reshape(-1).shape[0]
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)
        self.gamma = config['gamma']
        
        self.test_repeat = MAX_WORKERS * 2
        # Our two networks
        self.value_network = ValueNetwork(self.env.observation_space.shape[0], 16, 1)
        self.actor_network = ActorNetwork(self.env.observation_space.shape[0], 16, self.env.action_space.n)

        self.value_network2 = ValueNetwork2(input_dim, self.env2.ids.numAlert, 32, 1)
        self.actor_network2 = ActorNetwork2(input_dim, self.env2.ids.numAlert, 32, self.env2.ids.numAlert)
        
        # Their optimizers
        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(), lr=config['value_network']['learning_rate'])
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=config['actor_network']['learning_rate'])
        
        self.value_network_optimizer2 = optim.RMSprop(self.value_network2.parameters(), lr=config['value_network']['learning_rate'])
        self.actor_network_optimizer2 = optim.RMSprop(self.actor_network2.parameters(), lr=config['actor_network']['learning_rate'])
        
    # Hint: use it during training_batch
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
        episode_count = 0
        actions = np.empty((batch_size,), dtype=int)
        dones = np.empty((batch_size,), dtype=bool)
        rewards, values = np.empty((2, batch_size), dtype=float)
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=float)
        observation = self.env.reset()
        rewards_test = []
        
        belief = self.env2.reset()
        numAction2 = self.env2.ids.numAlert
        episode_count2 = 0
        actions2 = np.empty((batch_size,), dtype=int)
        dones2 = np.empty((batch_size,), dtype=bool)
        rewards2, values2 = np.empty((2, batch_size), dtype=float)
        observations2 = np.empty((batch_size,) + (belief.shape[0] + 1,), dtype=float)
        rewards_test2 = []
        action = 0

        for epoch in tqdm(range(epochs)):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                values[i] = self.value_network(torch.tensor(observation, dtype=torch.float)).detach().numpy()
                policy = self.actor_network(torch.tensor(observation, dtype=torch.float))
                actions[i] = torch.multinomial(policy, 1).detach().numpy()
                observation, rewards[i], dones[i], _ = self.env.step(actions[i])

                alert, _, _, _ = self.env2.evolve()
                observations2[i, :] = np.concatenate([belief, np.sum(2**alert).reshape(-1)], axis= 0)
                values2[i] = self.value_network2(torch.tensor([observations2[i]], dtype=torch.float)).detach().numpy()
                policy2 = self.actor_network2(torch.tensor([observations2[i]], dtype=torch.float))
                actions2[i] = np.squeeze(torch.multinomial(policy2, 1).detach().numpy())
                belief, rewards2[i], dones2[i] = self.env2.step(i, actions2[i])
                dones2[i] = batch_size - 1 == i
                if dones[i]:
                    observation = self.env.reset()

                    belief = self.env2.reset()

            # If our epiosde didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network(torch.tensor(observation, dtype=torch.float)).detach().numpy()[0]

            if dones2[-1]:
                next_value2 = 0
            else:
                alert, _, _, _ = self.env2.evolve()
                next_value2 = self.value_network2(torch.tensor([np.concatenate([belief, np.sum(2**alert).reshape(-1)], axis= 0)], dtype=torch.float)).detach().numpy()[0][0]
            
            # Update episode_count
            episode_count += sum(dones)

            episode_count2 += sum(dones2)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            returns2, advantages2 = self._returns_advantages(rewards2, dones2, values2, next_value2)

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages)

            self.optimize_model2(observations2, actions2, returns2, advantages2)

            # Test it every 50 epochs
            if epoch % 50 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(50)])) #TODO
                print(f'Sim1: Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')
                
                rng = default_rng()
                seedVector = rng.choice(9999999, size=self.test_repeat, replace=False)
                test = []
                with Pool(MAX_WORKERS) as p:
                    output = p.map(self.evaluate2, tqdm(np.random.randint(9999999, size=self.test_repeat)))
                # with Pool(MAX_WORKERS) as p:
                #     with tqdm(total=self.test_repeat) as pbar:
                #         for i, o in enumerate(p.imap_unordered(self.evaluate2, seedVector)):
                #             test.append(o)
                #             pbar.update()
                # rewards_test2.append(np.array(test))

                # rewards_test2.append(np.array([self.evaluate2(_) for _ in seedVector])) #TODO
                print(f'Sim2: Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test2[-1].mean(), 2)}, Std: {round(rewards_test2[-1].std(), 2)}')


                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs -1:
                    print('Early stopping !')
                    break
                observation = self.env.reset()
                belief = self.env2.reset()
                    
        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        
        r2 = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test2[i]) for i in range(len(rewards_test2))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r2, ci='sd');
        
        print(f'The trainnig was done over a total of {episode_count} episodes')
        print(f'The trainnig was done over a total of {episode_count2} episodes')

    def optimize_model(self, observations, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions), self.env.action_space.n)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)

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
        
        return loss_value, loss_actor    

    def optimize_model2(self, observations, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions), self.env2.ids.numAlert)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)

        # MSE for the values
        self.value_network_optimizer.zero_grad()
        values = self.value_network2(observations)
        loss_value = 1 * F.mse_loss(values, returns)
        loss_value.backward()
        self.value_network_optimizer.step()

        # Actor loss
        self.actor_network_optimizer.zero_grad()
        policies = self.actor_network2(observations)
        loss_policy = ((actions.float() * policies.log()).sum(-1) * advantages).mean()
        loss_entropy = - (policies * policies.log()).sum(-1).mean()
        loss_actor = - loss_policy - 0.0001 * loss_entropy
        loss_actor.backward()
        self.actor_network_optimizer.step()
        
        return loss_value, loss_actor    

    def evaluate(self, render=False):
        observation = self.env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False

        while not done:
            policy = self.actor_network(observation)
            action = torch.multinomial(policy, 1)
            observation, reward, done, info = self.env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode += reward            
        return reward_episode

    def evaluate2(self, seed):
        make_seed(seed)
        observation2 = self.env2.reset()
        observation2 = torch.tensor(observation2, dtype=torch.float)
        reward_episode2 = 0
        done2 = False
        i = 0
        
        while not done2:
            alert, _, _, _ = self.env2.evolve()
            observations2 = np.concatenate([observation2.reshape(-1), np.sum(2**alert).reshape(-1)], axis= 0)
            policy2 = self.actor_network2(torch.tensor([observations2], dtype=torch.float))
            action2 = np.squeeze(torch.multinomial(policy2, 1).detach().numpy())
            observation2, reward2, done2 = self.env2.step(i, action2)
            observation2 = torch.tensor(observation2, dtype=torch.float)
            reward_episode2 += reward2
            i += 1
            
        return reward_episode2