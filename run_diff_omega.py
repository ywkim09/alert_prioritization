#%%
from src.components.simulator import Simulator
import logging
import logging.handlers
import numpy as np
EPS = np.finfo(float).eps
import sys
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from src.components.CyberEnv import CyberEnv

# 2️⃣ Gym 환경 등록
def cyber_env_creator(env_config):
     return CyberEnv(env_config)

if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
    register_env("CyberSecurityEnv", cyber_env_creator)
    fp_rates = [None]
    w = 1
    t = "Inter" # t stands for attacker type. So, "Low" means a benign user / "Inter" means an attacker.
    num = "_two"
    for single_fp_rate in fp_rates: 
        config = {"runType": "My",
                "attackerType": t,
                "idsRatePath": f"config/IDS/idsRate{num}.json",
                "choicePath": f"config/attacker/attackChoice{num}.json",
                "successPath": f"config/attacker/attackSuccess{num}.json",
                "graphPath": f"config/environment/dependGraph{num}.json",
                "effectPath": f"config/environment/defenseEffect{num}.json",
                "testState": 0,
                "testDefenses": 0,
                "round": 50,
                "repeat": 32,
                "graph_verbose": False,
                "omega": [[0.0],[0.00001],[0.0001],[0.001],[0.01],[0.05],[0.1]],
                "parallel": True,
                "seed_val": 2,
                "untrained_actor_network" : "",
                "policy": ["No Investigation", "Bayes", "MinFP", "Random", "All"], #To add "RL", please train a model and add "RL" to the list.
                "epsilon": 0,
                "investigation_budget": [1],
                "sleep": 0,
                "fp_rate" : single_fp_rate,
                "state_threshold": float(sys.argv[1]), #0.001
                "exploit_threshold": float(sys.argv[2]), #0.0
                "run_ID": 0,
                }
        simulator = Simulator(**config)
        if 'RL' in config['policy']:
                RL_PPO_config = (
                PPOConfig()
                .environment(env="CyberSecurityEnv", env_config=config)
                .framework("torch")
                .env_runners(num_env_runners=1)   
                .training(
                        gamma=0,
                        kl_coeff=0.3,
                        train_batch_size_per_learner=256,
                        lr=0.00001,
                        model={
                        "fcnet_hiddens": [128, 64, 32],
                        "fcnet_activation": "LeakyReLU"
                        }
                )
                .resources(num_gpus=0, num_cpus_per_worker=1)
                )
                trainer = RL_PPO_config.build()
                trainer.restore('path/to/checkpoint')  # Load your trained model checkpoint here
                policy = trainer.learner_group._learner.module
                policy_model = policy['default_policy']
                # policy_model.forward({"obs":torch.rand(184)})['action_dist_inputs'].sample()


                simulator.add_RL_model(policy_model)
        simulator.runSim()
##%%
# %%
