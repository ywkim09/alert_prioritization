#%%
from src.components.simulator import Simulator
import logging
import logging.handlers
import numpy as np
EPS = np.finfo(float).eps
import sys

if __name__ == '__main__':
    fp_rates = [[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0.3,	0.2],
                [0.1,	0.05,	0.1,	0.05,	0,	0.05,	0,	0,	0.1,	0.4,	0.3],
                [0.2,	0.15,	0.2,	0.15,	0.1,	0.15,	0.1,	0.05,	0.2,	0.5,	0.4],
                [0.3,	0.25,	0.3,	0.25,	0.2,	0.25,	0.2,	0.15,	0.3,	0.6,	0.5],
                [0.4,	0.35,	0.4,	0.35,	0.3,	0.35,	0.3,	0.25,	0.4,	0.7,	0.6],
                [0.5,	0.45,	0.5,	0.45,	0.4,	0.45,	0.4,	0.35,	0.5,	0.8,	0.7],
                [0.6,	0.55,	0.6,	0.55,	0.5,	0.55,	0.5,	0.45,	0.6,	0.9,	0.8],
                [0.7,	0.65,	0.7,	0.65,	0.6,	0.65,	0.6,	0.55,	0.7,	1,	0.9],
                [0.8,	0.75,	0.8,	0.75,	0.7,	0.75,	0.7,	0.65,	0.8,	1,	1],
                [0.9,	0.85,	0.9,	0.85,	0.8,	0.85,	0.8,	0.75,	0.9,	1,	1],
                [1,	0.95,	1,	0.95,	0.9,	0.95,	0.9,	0.85,	1,	1,	1]]
    w = 1
    t = "Inter"
    num = "_small"
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
                "omega": [[0.05], [0]],
                "paraller": True,
                "actor_network" : "",
                "seed_val": 2,
                "untrained_actor_network" : "",
                "policy": ["No Investigation", "Bayes", "MaxEntropy", "MinFP", "Random", "All"],
                "epsilon": EPS,
                "investigation_budget": [1],
                "sleep": 0,
                "fp_rate" : single_fp_rate,
                "state_threshold": float(sys.argv[1]), #0.001
                "exploit_threshold": float(sys.argv[2]), #0.45
                "run_ID": sys.argv[3]
                }

        simulator = Simulator(**config)
        simulator.runSim()
##%%
# %%
