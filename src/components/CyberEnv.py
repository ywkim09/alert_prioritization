
import gymnasium as gym
from gymnasium import spaces

import numpy as np
try:
    from src.components.simulator import Simulator
except:
    import os, sys, inspect
    from pathlib import Path
    from src.components.simulator import Simulator

EPS = np.finfo(float).eps
class CyberEnv(gym.Env): 
    """
    OpenAI Gym environment wrapper for the custom CyberSecurity Simulator.
    """
    def __init__(self, config):
        super(CyberEnv, self).__init__()

        # Initialize the custom simulator
        self.simulator = Simulator(**config)
        
        self.action_space = spaces.Discrete(self.simulator.action_space)
        
        # Observations include the belief state and alert vector
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.simulator.observation_space_size,),
            dtype=np.float32
        )
        # Initialize state
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        if seed is None:
            seed = np.random.randint(0, 9999999)

        self.state = self.simulator.reset(seed)
        

        # 초기화
        observation = np.array(self.state, dtype=np.float32)
        info = {
            "seed": seed,
            "episode_reward": 0.0,  
            "step_count": 0         
        }
        
        self.total_reward = 0.0  
        self.step_count = 0

        return observation, info

    def step(self, action):
        """
        Execute one time step within the environment.
        """

        next_state, reward, done = self.simulator.step(action, self.step_count)
        # print(f"[DEBUG] Step Result - State: {next_state}, Reward: {reward}, Done: {done}")
        self.step_count += 1
        self.total_reward += reward
        terminated = done
        truncated = done  

        info = {
            "episode_reward": self.total_reward,
            "step_count": self.step_count,
            "terminated": terminated,
            "truncated": truncated
        }
        return (
            np.array(next_state, dtype=np.float32), 
            reward, 
            terminated, 
            truncated, 
            info
        )


    def render(self, mode='human'):
        """
        Rendering logic, currently not implemented.
        """
        pass

    def close(self):
        """
        Environment closing logic.
        """
        pass
