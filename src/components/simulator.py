import numpy as np
import os
import sys
from src.components.ids import IDS, MyIDS
from src.components.graph import Graph
from src.components.attacker import Attacker
from src.components.defender import Defender, MyDefender
from src.util.auxiliaries import folderCreation
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import torch
import pickle
import copy
from src.rl.model import ActorNetwork2
import mpltex
import time
EPS = np.finfo(float).eps

MAX_WORKERS = 32

def extractKeys(input):
    output = {}
    depth = 0
    for k, v in input.items():
        if isinstance(v, dict) is True:
            depth +=1
            if output.get(k) is None:
                output[k] = {}
            output[k] = v.keys()
        elif isinstance(v, list) is True:
            return {k: "Done"}
    return output
        
class Simulator:
    def __init__(self, runType, attackerType, idsRatePath, choicePath, successPath, graphPath, 
    effectPath, testState, testDefenses, round, repeat, graph_verbose, omega, paraller, policy, seed_val, investigation_budget, sleep,
    state_threshold, exploit_threshold, run_ID, epsilon = 10**(-100), fp_rate = [0,0,0,0,0,0,0,0], actor_network ='', untrained_actor_network =''):
        self.seed_val = seed_val
        self.set_seed(self.seed_val)
        self.runType = runType
        self.attackerType = attackerType
        self.idsRatePath = idsRatePath
        self.choicePath = choicePath
        self.successPath = successPath
        self.effectPath = effectPath
        self.graphPath = graphPath
        self.testState = testState
        self.testDefenses = testDefenses
        self.round = round
        self.repeat = repeat
        self.classIDS = MyIDS if self.runType == "Basic" else MyIDS
        self.classDefender = Defender if self.runType == "Basic" else MyDefender
        self.classGraph = Graph
        self.classAttacker = Attacker 
        self.graph = None
        self.figs = {}
        self.initialBeliefFig = None
        self.graph_verbose = graph_verbose
        self.confidence = 1 - np.array(omega)
        self.confidence[self.confidence==0] = EPS
        self.confidence[self.confidence==1] = 1-EPS
        self.paraller = paraller
        self.policy = policy
        self.repeat_result = {}
        self.final_result = {}
        self.toSave = ['Figure', 'MSE', 'Entropy']
        self.graphSave = ['MSE', 'Entropy']
        self.epsilon = epsilon
        self.state = 0
        self.y_t = np.array([], dtype=int)
        self.E_t = np.array([], dtype=int)
        self.y_t_true = np.array([], dtype=int)
        self.i = 0
        self.ib = investigation_budget
        self.state_threshold = state_threshold
        self.exploit_threshold = exploit_threshold
        self.run_ID = run_ID
        self.sleep = sleep
        self.fp_rate = fp_rate
        self.ids            = self.classIDS(self.attackerType, self.idsRatePath, self.fp_rate, 0) 
        self.graph          = self.classGraph(self.ids, self.attackerType, self.graphPath, self.effectPath, self.epsilon, 
                                            testState = self.testState, testDefense = self.testDefenses, graph_verbose=self.graph_verbose)
        self.observation_space = np.zeros(self.graph.feasibleStates.shape[0] * self.graph.numAttType + 1) # self.beliefRL.shape[0] * self.beliefRL.shape[1]
        self.action_space = self.graph.numAlerts + 1 # self.numAlerts + 1 
        self.actor_network_dict = {}
        self.actor_network_dict['training'] = ActorNetwork2(self.observation_space.shape[0], self.action_space, 32)
        self.actor_network_dict['untrained'] = copy.deepcopy(self.actor_network_dict['training'])
        if os.path.isfile(actor_network):
            self.actor_network_dict['training'].load_state_dict(torch.load(actor_network))
        if os.path.isfile(untrained_actor_network):
            self.actor_network_dict['untrained'].load_state_dict(torch.load(untrained_actor_network))

    def set_seed(self, seed):
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

    def initResultDict(self, input):
        if len(input.keys()) == 0:
            pass
        else:
            for key1, value1 in input.items():
                if type(value1) == dict:
                    input[key1] = self.initResultDict(value1)
                else:
                    input[key1] = []
        return input

    def saveResultDict(self, input, output, metrics):
        if len(output.keys())==0:
            for key in metrics:
                output[key] = {}
                for key2 in input[key].keys():
                    output[key][key2]= []
        
        for key, value1 in output.items():
            for policy, value2 in value1.items():
                output[key][policy].append(input[key][policy])
        return output

    def reset(self, seed):
        self.saveKeys       = self.graphSave if self.paraller else self.toSave 
        self.repeat_result  = self.initResultDict(self.repeat_result)
        self.ids            = self.classIDS(self.attackerType, self.idsRatePath, self.fp_rate, seed) 
        self.graph          = self.classGraph(self.ids, self.attackerType, self.graphPath, self.effectPath, self.epsilon, 
                                            testState = self.testState, testDefense = self.testDefenses, graph_verbose=self.graph_verbose)
        self.attacker       = self.classAttacker(self.ids, self.graph, self.attackerType, self.choicePath, self.successPath, seed, self.testState)
        self.defender       = self.classDefender(self.ids, self.graph, self.choicePath, self.successPath, self.policy, self.epsilon, seed,
                                            exploit_threshold= self.exploit_threshold, state_threshold = self.state_threshold, ib = self.ib, actor_network_dict = self.actor_network_dict, testDefense = self.testDefenses, omega = self.confidence)
        self.i = 0
        self.state = 0
        self.y_t = np.array([], dtype=int)
        self.E_t = np.array([], dtype=int)
        self.y_t_true = np.array([], dtype=int)
        return np.concatenate([self.defender.beliefRL.reshape(-1), np.sum(2**self.y_t).reshape(-1)], axis= 0)

    def evolve(self, r):
        E_t = np.array([], dtype=np.int8)
        E_t_possible = np.array([], dtype=np.int8)
        if r >= self.sleep:
            E_t, E_t_possible, E_t_blocked            = self.attacker.chooseExploits()
        self.y_t, self.y_t_true, y_t_false        = self.ids.genAlerts(E_t)
        s_iʼ, E_t_succeed                         = self.attacker.performExploits(E_t_possible)
        return self.y_t, self.y_t_true, E_t, s_iʼ

    def step(self, action):
        belief, investigation                     = self.defender.oneUpdate(action, self.y_t, self.y_t_true)
        reward                                    = self.graph.calReward(self.state, belief, self.y_t, investigation)
        mse_reward                                = reward['MSE']
        entropy_reward                            = reward['Entropy']
        y_t, y_t_true, E_t, self.state            = self.evolve()
        observation = np.concatenate([belief.reshape(-1), np.sum(2**y_t).reshape(-1)], axis= 0)
        done = self.i >= self.round - 1
        self.i += 1 
        return observation, mse_reward, done

    def saveData(self, newBelief, newState, alerts_n, trueAlerts, chosenExploit):
        result = {}
        for id in self.toSave:
            result[id]= {}

        for key in result:
            for policy in newBelief.keys():
                result[key][policy] = None
 
        for policy, info in newBelief.items():
            belief, alerts_k = info
            output = self.graph.drawGraph(newState, alerts_n, trueAlerts, chosenExploit, belief, alerts_k=alerts_k)
            for key, value in output.items():
                result[key][policy] = value    
        return result

    def simulator(self, seed):
        start = time.time()
        self.reset(seed)
        self.set_seed(seed) 
        for r in range(self.round):
            # fig, mse, entropy   = self.graph.drawGraph(newState, alerts_n, chosenExploit, self.defender.belief)
            if r == 0:
                pi_t_1                                           = self.defender.skip()
            else:
                pi_t_1                                           = self.defender.updateBelief(self.y_t, self.y_t_true, self.state, self.y_t, self.y_t_true, self.E_t)
            result                                           = self.saveData(pi_t_1, self.state, self.y_t, self.y_t_true, self.E_t)
            self.repeat_result                               = self.saveResultDict(result, self.repeat_result, self.saveKeys)
            self.y_t, self.y_t_true, self.E_t, self.state    = self.evolve(r)
        end = time.time()
        return self.repeat_result, self.defender.exploit_threshold, end - start, self.defender.time_dict
        
    def saveResults(self):
        subfolder_name = str(self.repeat)+ "_runs_" + self.run_ID
        self.folder = folderCreation(os.path.join("result", subfolder_name))
        fp_rate = np.array(self.ids.idsFPRate).mean()
        with open(os.path.join(self.folder, f'result_{fp_rate}_{self.ib}.pkl'), 'wb') as f:
            pickle.dump(self.final_result, f)

        with open(os.path.join(self.folder, f'time_dict.pkl'), 'wb') as f:        
            pickle.dump(self.t_out_dict, f)
        if self.graph_verbose and not self.paraller:
            for k, figArray in self.repeat_result["Figure"].items():
                for repeat, figList in enumerate(figArray):
                    figList.savefig(os.path.join(self.folder, f"{k}_{repeat}.round.png"), format="PNG")

        save_dict = {}
        for metric, v in self.final_result.items():
            for k2, v2 in v.items():
                policy = k2.split('_')[0] 
                omega = k2.split('_')[1]
                save_dict[omega] = {}
        for metric, v in self.final_result.items():
            for k,v in save_dict.items():
                save_dict[k][metric] = {}

        for metric, v in self.final_result.items():
            for k2, v2 in v.items():
                policy = k2.split('_')[0] 
                omega = k2.split('_')[1]
                try:
                    save_dict[omega][metric][policy] = v2
                except:
                    save_dict[omega][metric] = {}
                    save_dict[omega][metric][policy] = v2

        linestyles = mpltex.linestyle_generator(colors=[],
                                        lines=['-',':'],
                                        markers=['o','^', 'x'],
                                        hollow_styles=[False, True],)

        fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(20,5))

        i = 0
        for metric, metric_dict in save_dict[list(save_dict.keys())[0]].items():
            for policy, rollouts in metric_dict.items():
                ax[i].plot(rollouts.mean(axis=0), 
                            lw=2,
                            label=f'{policy}', 
                            **next(linestyles),
                            markevery=5
                            )
                ax[i].set_title('')
            ax[i].legend(loc="upper left")
            ax[i].set_xlabel('Step (t)', labelpad=10)
            if i == 0:
                ax[i].set_ylabel('MSE', labelpad=10)
            else:
                ax[i].set_ylabel('Entropy', labelpad=10)
            i += 1

        fig.savefig('mse_entropy.eps', format='eps', bbox_inches='tight')


        policy_dict = {}
        for policy_omega, rollouts in self.final_result['MSE'].items():
            try:
                policy_dict[policy_omega.split('_')[0]][policy_omega] = rollouts
            except:
                policy_dict[policy_omega.split('_')[0]] = {}
                policy_dict[policy_omega.split('_')[0]][policy_omega] = rollouts
                
        re = {}
        for policy, rollouts in policy_dict.items():
            output = []
            for k,v in rollouts.items():
                output.append([1-float(k.split('_')[1]), v[:,-1].mean()])
            coordinates = np.array(output)
            sorted_coord = coordinates[coordinates[:, 0].argsort()]
            re[k.split('_')[0]] = sorted_coord

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        for key, val in re.items():
            ax2.plot(val[:,0], 
                        val[:,1], 
                        lw=2,
                        label=f'{key}', 
                        **next(linestyles),
                        markevery=1
                        )
        ax2.set_xscale('log')
        ax2.legend(loc="lower right")
        ax2.set_xlabel('Uncertainty of investigation ($log(1-\omega)$)', labelpad=10)
        ax2.set_ylabel('MSE', labelpad=10)
        fig2.savefig('1-omega_vs_mse.eps', format='eps', bbox_inches='tight')
        
    def runSim(self):
        rng = np.random.default_rng(seed=self.seed_val)
        seedVector = rng.choice(9999999, size=self.repeat, replace=False)
        if self.paraller: 
            output = []
            e_ths = []
            time_lst = []
            time_dict_lst = []
            with Pool(MAX_WORKERS) as p:
                with tqdm(total=self.repeat) as pbar:
                    for i, o in enumerate(p.imap_unordered(self.simulator, seedVector)):
                        output.append(o[0])
                        e_ths.append(o[1])
                        time_lst.append(o[2])
                        time_dict_lst.append(o[3])
                        pbar.update()
            for t in output:
                self.final_result = self.saveResultDict(t, self.final_result, self.graphSave)
        else:
            e_ths = []
            time_lst = []
            time_dict_lst = []
            for seed in tqdm(seedVector):
                repeat_result = self.simulator(seed)
                self.final_result = self.saveResultDict(repeat_result[0], self.final_result, self.graphSave)
                e_ths.append(repeat_result[1])              
                time_lst.append(repeat_result[2])      
                time_dict_lst.append(repeat_result[3])            
        for key, value in self.final_result.items():
            for policy, value2 in value.items():
                self.final_result[key][policy] = np.array(self.final_result[key][policy])
        self.t_out_dict = {}
        for p in self.policy:
            self.t_out_dict[p] = []
            for time_dict in time_dict_lst:
                self.t_out_dict[p].append(time_dict[p])
        self.saveResults()
        path = os.path.join(self.folder, 'readme.txt')
        with open(path, 'w') as f:
            f.write(f'repeat: {self.repeat}\n')
            f.write(f"state threshold: {self.state_threshold} \n")
            f.write(f"exploit threshold: {self.exploit_threshold} \n")
            f.write(f'time_lst: {time_lst}\n')
    