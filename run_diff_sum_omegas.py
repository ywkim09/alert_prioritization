#%%
from src.components.simulator import Simulator
import logging
import logging.handlers
import numpy as np
EPS = np.finfo(float).eps
import sys


if __name__ == '__main__':
    omega_lsts = [[[0	, 0.3],
                [0.0375	, 0.2625],
                [0.075	, 0.225],
                [0.1125	, 0.1875],
                [0.15	, 0.15]],
             [[0	, 0.4],
                [0.05	, 0.35],
                [0.1	, 0.3],
                [0.15	, 0.25],
                [0.2	, 0.2]]]
    w = 1
    t = "Inter"
    num = "_two"
    for omega in omega_lsts:
        print(w, num, t)
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
                "repeat": 200,
                "graph_verbose": False,
                "omega": omega,
                "paraller": True,
                "actor_network" : "",
                "seed_val": 2,
                "untrained_actor_network" : "",
                "policy": ["No Investigation", "Bayes", "MaxEntropy", "MinFP", "Random", "All"],
                "epsilon": 0,
                "investigation_budget": [1,1],
                "sleep": 0,
                "fp_rate" : [0.45, 0.5, 0.45, 0.5, 0.45, 0.6, 0.55, 0.5],
                "state_threshold": float(sys.argv[1]), #0.001
                "exploit_threshold": float(sys.argv[2]), #0.45
                "run_ID": sys.argv[3]
                }
        simulator = Simulator(**config)
        simulator.runSim()
##%%