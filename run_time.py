#%%
from src.components.simulator import Simulator
import logging
import logging.handlers
import numpy as np
import sys

if __name__ == '__main__':
    fp_rates = [None]
    w = 1
    t = "Inter"
    num = "_real"
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
                "omega": [[0]],
                "paraller": True,
                "actor_network" : "",
                "seed_val": 2,
                "untrained_actor_network" : "",
                "policy": ["No Investigation", "Bayes", "MaxEntropy", "MinFP", "Random", "All"],
                "epsilon": 10**(-100),
                "investigation_budget": 1,
                "sleep": 0,
                "fp_rate" : single_fp_rate,
                "state_threshold": float(sys.argv[1]), #0.001
                "exploit_threshold": float(sys.argv[2]), #0.45
                "run_ID": sys.argv[3]
                }
        simulator = Simulator(**config)
        print('run')
        simulator.runSim()
##%%
# %%
