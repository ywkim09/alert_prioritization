import json
from more_itertools import powerset
import numpy as np
import pandas as pd
import networkx as nx
import pylab
import matplotlib.pyplot as plt
from scipy.special import entr
from src.components.ids import IDS, MyIDS
try:
    from src.util.auxiliaries import checkAttackerType, convKeyInt, convertDec2Bin, powerSet
except:
    import os, sys
    p = os.path.abspath('.')
    sys.path.insert(1, p)
    from src.util.auxiliaries import checkAttackerType, convKeyInt, convertDec2Bin, powerSet

ERRORBOUND = 10**-14

class Graph:
    def __init__(self, ids: MyIDS , attackerType, graphPath, effectPath, epsilon, testState = 0, testDefense = 0, graph_verbose= False):
        self.curDefenses = testDefense
        self.curState = testState
        self.ids = ids
        with open(graphPath, 'r') as f:
            graph = json.load(f)
        with open(effectPath, 'r') as f:
            effect = json.load(f)
        self.attackerType = attackerType
        self.type2int = self.ids.type2int
        self.int2type = {v:k for k,v in self.type2int.items()}
        # graph = checkAttackerType(attackerType, graph)
        self.graph = convKeyInt(graph)
        # effect = checkAttackerType(attackerType, effect)
        self.effect = convKeyInt(effect)
        self.numDefense = len(self.effect.keys()) 
        self.numCondition = len(self.graph["Condition"].keys())
        self.numExploit = len(self.graph["PreCondition"].keys())
        self.numAttType = self.ids.idsFPMatrix.shape[0]
        self.numAlerts = self.ids.numAlert
        self.goalState = np.array(self.graph["GoalState"])
        self.preCondition = np.zeros((self.numExploit, self.numCondition), dtype=np.int64)
        self.postCondition = np.zeros((self.numExploit, self.numCondition), dtype=np.int64)
        self.conditions = np.arange(self.numCondition)
        self.exploits = np.arange(self.numExploit)
        self.defenses = np.arange(self.numDefense)
        for key in sorted(self.graph["PreCondition"].keys()):
            for entry in self.graph["PreCondition"][key]:
                self.preCondition[key-1][entry-1] = 1
        for key in sorted(self.graph["PostCondition"].keys()):
            for entry in self.graph["PostCondition"][key]:
                self.postCondition[key-1][entry-1] = 1
        self.defenseEffect = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in self.effect.items() ])).fillna(-1).astype(int).T
        self.defenseEffect = self.defenseEffect.values - 1
        self.feasibleStates, self.feasibleExploits = self.getFeasibleStates()
        self.defensedExploits = np.array([], dtype=int)
        self.idxFeasibleStates = np.zeros(2**self.numCondition, dtype=int)
        for i, v in enumerate(self.feasibleStates):
            self.idxFeasibleStates[v] = i
        self.graph_verbose = graph_verbose
        with open(graphPath, 'r') as f:
            graph = json.load(f)
        graph = convKeyInt(graph)
        self.G = nx.DiGraph()
        self.epsilon = epsilon
        attType = np.zeros(self.numAttType)
        attType[self.type2int[self.attackerType]] = 1
        self.mse     = np.sum((attType - (np.ones(self.numAttType) * 1/self.numAttType))**2)
        self.entropy = entr(np.ones(self.numAttType) * 1/self.numAttType).sum()/np.log(2)
        type_pos = 5 - (len(self.int2type.keys())-1)/2
        self.n_pos   = [graph["ConditionPosition"][k] for k in sorted(graph["ConditionPosition"].keys())]
        self.e_pos   = [graph["ExploitPosition"][k] for k in sorted(graph["ExploitPosition"].keys())]
        self.t_pos = []
        for key in sorted(self.int2type.keys()):
            self.t_pos.append([0, type_pos])
            type_pos += 1
        self.impossible_reward = -100
        self.lower_bound = -100
        self.multi_factor = 10
        self.nothing_reward = 10
        self.valid_reward = 1
        self.labels = {}
        
        idx = 0
        for p in self.n_pos:
            self.G.add_node(idx, s="o", c='red', z = 200, pos = p)
            self.labels[idx]=  '' 
            idx += 1
        for p in self.e_pos:
            self.G.add_node(idx, s="h", c='red', z = 100, pos = p)
            self.labels[idx]= ''
            idx += 1
        for p in self.t_pos:
            self.G.add_node(idx, s="s", c='red', z = 100, pos = p)
            self.labels[idx]= ''
            idx += 1

        for k, v in graph['PreCondition'].items():
            for v2 in v:
                node = v2 - 1
                exploit = k + self.numCondition - 1
                self.G.add_edge(node, exploit)

        for k, v in graph['PostCondition'].items():
            for v2 in v:
                node = v2 - 1
                exploit = k + self.numCondition - 1
                self.G.add_edge(exploit, node)

        self.nodePos = nx.get_node_attributes(self.G, 'pos')
        # self.nodePos = nx.spring_layout(self.G)

    def checkAvailExploit(self, state):
        binState = convertDec2Bin(state, self.numCondition)
        satisfyPre = (np.squeeze(np.tile(binState, (self.numExploit, 1)).reshape([self.numExploit, *binState.shape]).transpose(1,0,2)) - self.preCondition) >=0
        boolPre = np.all(satisfyPre, axis=-1)
        satisfyPost = (self.postCondition - np.squeeze(np.tile(binState, (self.numExploit, 1)).reshape([self.numExploit, *binState.shape]).transpose(1,0,2))) > 0
        boolPost = np.any(satisfyPost, axis=-1)
        selectedExploit = boolPre & boolPost
        availExploit = np.squeeze(np.tile(self.exploits,(binState.shape[0],1)))
        availExploit[np.invert(selectedExploit)] = -1
        return availExploit
    
    def checkAvailExploit_bin(self, binState):
        satisfyPre = (np.tile(binState, (self.numExploit, 1)) - self.preCondition) >=0
        boolPre = np.all(satisfyPre, axis=-1)
        satisfyPost = (self.postCondition - np.tile(binState, (self.numExploit, 1))) > 0
        boolPost = np.any(satisfyPost, axis=-1)
        selectedExploit = boolPre & boolPost
        availExploit = np.arange(self.numExploit) # np.tile(self.exploits,(binState.shape[0],1))
        availExploit[np.invert(selectedExploit)] = -1
        return availExploit

    def checkBlockedExploit(self, availExploit):
        binDefenses = convertDec2Bin(self.curDefenses, self.numDefense) > 0
        defensedExploits = np.expand_dims(self.defenseEffect, axis=0)[binDefenses]
        self.defensedExploits = np.unique(defensedExploits).reshape(-1)
        blockedExploits = np.array(np.intersect1d(availExploit, self.defensedExploits), dtype = int)
        return blockedExploits

    def updateState(self, successExploit):
        s_i_bin = convertDec2Bin(self.curState, self.numCondition)
        s_i_plus = np.sum(self.postCondition[successExploit], axis = 0)
        s_iʼ = (s_i_bin + s_i_plus) > 0
        self.curState = s_iʼ.dot(2**np.arange(s_iʼ.size))
        return self.curState
    
    def updateDefense(self, chosenDefense):
        binDefense = (convertDec2Bin(self.curDefenses, self.numDefense) > 0).reshape(-1)
        if chosenDefense == 0:
            pass
        else:    
            binDefense[chosenDefense - 1] = True
        self.defensedExploits = np.unique(self.defenseEffect[binDefense][self.defenseEffect[binDefense] >= 0].reshape(-1))
        self.curDefenses = binDefense.dot(2**np.arange(binDefense.size))
        return self.curDefenses, self.defensedExploits
 
    def getFeasibleStates(self):
        feasibleStates = [np.array([0])]
        self.feasibleExploits = {} 
        for state in feasibleStates:
            curState = convertDec2Bin(state, self.numCondition)
            availExploits = self.checkAvailExploit(state)
            availExploits = availExploits[availExploits>=0]
            for exploits in powerSet(availExploits):
                if len(exploits) != 0:
                    feasibleState = np.array((curState + np.sum(self.postCondition[np.array(exploits)], axis=0))> 0, dtype=int)
                    feasibleState = feasibleState.dot(2**np.arange(feasibleState.size))
                    if feasibleState not in feasibleStates:
                        feasibleStates.append(feasibleState)
        self.feasibleStates = np.sort(np.array(feasibleStates).reshape(-1))
        for state in self.feasibleStates:
            availExploits = self.checkAvailExploit(state)
            availExploits = availExploits[availExploits>=0]
            self.feasibleExploits[state] = availExploits
        return self.feasibleStates, self.feasibleExploits

    def drawGraph(self, state, alerts_n, trueAlerts, exploit, belief, alerts_k=None):
        #Get all distinct node classes according to the node shape attribute
        output = {}
        binState = convertDec2Bin(state, self.numCondition)
        alpha = {}
        belief = belief.copy()
        belief[belief==self.epsilon] = 0
        beliefStates = np.sum(belief, axis=1)
        states = np.zeros(self.numCondition)
        feasibleStatesBin = convertDec2Bin(self.feasibleStates, self.numCondition)
        for i, s in enumerate(feasibleStatesBin):
            conditions = s > 0
            states[conditions] += beliefStates[i]
        for i, v in enumerate(states):
            states[i]= min(v, 1.0)

        alpha['o'] = states
        alpha['h'] = np.zeros(self.numExploit)
        beliefType = np.sum(belief, axis=0)
        alpha['s'] = beliefType
        if np.any(beliefType > 1 + ERRORBOUND):
            print(f"{max(beliefType)} is bigger than the error bound of float 64.")
        else:
            alpha['s'] = np.minimum(beliefType, np.ones(beliefType.shape))
        color = {}
        for k, v in alpha.items():
            color[k] = ['w' if alpha == 0.0 else 'r' for alpha in v]
        if self.graph_verbose:
            nodeShapes = set([aShape[1]['s'] for aShape in self.G.nodes(data = True)])
            fig = plt.figure()
            for i in self.labels.keys():
                self.labels[i] = ''
            for i in np.expand_dims(np.arange(self.numCondition), axis=0)[binState>0]:
                self.labels[i] = '●'
            for i in exploit:
                self.labels[i + self.numCondition] = '⬢'
            for i in self.defensedExploits:
                self.labels[i+ self.numCondition] = 'X'
            self.labels[self.numCondition + self.numExploit + self.type2int[self.attackerType]] = '■'

            for shape in nodeShapes: 
                nx.draw_networkx_nodes(
                        self.G, 
                        self.nodePos, 
                        node_shape = shape, 
                        node_color = color[shape],
                        node_size = [sNode[1]["z"] for sNode in filter(lambda x: x[1]["s"]==shape, self.G.nodes(data = True))],
                        nodelist = [sNode[0] for sNode in filter(lambda x: x[1]["s"]==shape, self.G.nodes(data = True))],
                        linewidths = 1,
                        edgecolors = 'black',
                        alpha = alpha[shape]
                        )
            nx.draw_networkx_edges(self.G, self.nodePos, arrows =True)
            nx.draw_networkx_labels(self.G, self.nodePos, self.labels, font_size=5, font_color="black")
            alertVec_n = np.zeros(self.numAlerts, dtype=np.int8)
            alertVec_n[alerts_n] = 1

            alertVec_t = np.zeros(self.numAlerts, dtype=np.int8)
            alertVec_t[trueAlerts] = 1
            
            plt.figtext(0.5, 0.09, f"Alert  : {alertVec_n}", wrap=True, horizontalalignment='center', fontsize=8)
            plt.figtext(0.5, 0.05, f"True   : {alertVec_t}", wrap=True, horizontalalignment='center', fontsize=8)
            
            if alerts_k is not None:
                plt.figtext(0.5, 0.01, f"Invest: {alerts_k}", wrap=True, horizontalalignment='center', fontsize=8)
            pylab.close()
            output['Figure'] = fig
        
        attTypeVec = np.zeros(self.numAttType)
        attTypeVec[self.type2int[self.attackerType]] = 1
        mse = self.calMSE(alpha['o'], binState)
        # mse = self.calMSE_ct(alpha['o'], binState, alpha['s'], attTypeVec)
        entropy = entr(belief).sum()/np.log(2)
        output['MSE'] = mse
        output['Entropy'] = entropy
        return output
        
    def calMSE(self, conditions, binState):
        mse = (((conditions - binState)**2).sum() )/(self.numCondition)
        return mse

    def calMSE_ct(self, conditions, binState, types, attTypeVec):
        mse = (((conditions - binState)**2).sum() + ((types - attTypeVec)**2).sum())/(self.numCondition + self.numAttType)
        return mse

    ########## TEST FUNCTION #############
    def checkPossibleTransition(self, currentState):
        availExploit = self.checkAvailExploit(currentState)
        availExploit = availExploit[availExploit>=0]
        blockedExploit = self.checkBlockedExploit(availExploit)
        possibleExploit = np.setdiff1d(availExploit, blockedExploit)
        return possibleExploit, blockedExploit

    def calReward(self, state, belief, y_t, alert_k=None):
        #Get all distinct node classes according to the node shape attribute
        output = {}
        binState = convertDec2Bin(state, self.numCondition)
        alpha = {}
        belief = belief.copy()
        belief[belief==self.epsilon] = 0
        beliefStates = np.sum(belief, axis=1)
        conditions = np.zeros(self.numCondition)
        feasibleStatesBin = convertDec2Bin(self.feasibleStates, self.numCondition)
        for i, s in enumerate(feasibleStatesBin):
            active_condtions = s > 0
            conditions[active_condtions] += beliefStates[i]
        for i, v in enumerate(conditions):
            conditions[i]= min(v, 1.0)
        alpha['o'] = conditions
        mse = self.calMSE(alpha['o'], binState)
        entropy = entr(belief).sum()/np.log(2)

        if len(y_t) == 0 and alert_k[0] >= self.numAlerts:
            output['MSE'] = self.nothing_reward
            output['Entropy'] = self.nothing_reward
        elif alert_k[1] is not None:
            mse_reward = max((self.mse - mse) * self.multi_factor / (self.mse + ERRORBOUND), self.lower_bound)
            output['MSE'] = mse_reward + self.valid_reward
            output['Entropy'] = (self.entropy - entropy) * self.multi_factor / self.entropy + self.valid_reward
        else:
            output['MSE'] = self.impossible_reward
            output['Entropy'] = self.impossible_reward

        self.mse = mse
        self.entropy = entropy
        #         #Get all distinct node classes according to the node shape attribute
        # output = {}
        # if len(y_t) == 0 and alert_k[0] >= self.numAlerts:
        #     output['MSE'] = self.nothing_reward
        #     output['Entropy'] = self.nothing_reward
        # elif alert_k[1] is not None:
        #     binState = convertDec2Bin(state, self.numCondition)

        #     alpha = {}
        #     belief = belief.copy()
        #     belief[belief==self.epsilon] = 0
        #     beliefStates = np.sum(belief, axis=1)
        #     states = np.zeros(self.numCondition)
        #     feasibleStatesBin = convertDec2Bin(self.feasibleStates, self.numCondition)
        #     for i, s in enumerate(feasibleStatesBin):
        #         conditions = s > 0
        #         states[conditions] += beliefStates[i]
        #     for i, v in enumerate(states):
        #         states[i]= min(v, 1.0)

        #     alpha['o'] = states
        #     beliefType = np.sum(belief, axis=0)
        #     alpha['s'] = beliefType
        #     if np.any(beliefType > 1 + ERRORBOUND):
        #         print(f"{max(beliefType)} is bigger than the error bound of float 64.")
        #     else:
        #         alpha['s'] = np.minimum(beliefType, np.ones(beliefType.shape))
        #     attTypeVec = np.zeros(self.numAttType)
        #     attTypeVec[self.type2int[self.attackerType]] = 1  
        #     # mse = self.calMSE(alpha['o'], binState)
        #     mse = self.calMSE_ct(alpha['o'], binState, alpha['s'], attTypeVec)
        #     entropy = entr(belief).sum()/np.log(2)
        #     output['MSE'] = (self.mse - mse) * 10 / (self.mse + ERRORBOUND) 
        #     output['Entropy'] = (self.entropy - entropy) * 10 / self.entropy
        #     self.mse = mse + self.valid_reward
        #     self.entropy = entropy + self.valid_reward
        # else:
        #     output['MSE'] = self.impossible_reward
        #     output['Entropy'] = self.impossible_reward

        return output

if __name__ == "__main__":
    attackerType = "Inter"
    # IDS
    idsRatePath = "IDS/idsRate.json"
    # Attacker 
    choicePath = "attacker/attackChoice.json"
    successPath = "attacker/attackSuccess.json"
    # Defender
    detectPath = "defender/defendDetect.json"
    effectPath = "environment/defenseEffect.json"
    # Graph
    # effectPath = "environment/defenseEffect.json"
    graphPath = "environment/dependGraph.json"
    
    testState = 15
    testDefenses = 3

    graph = Graph(attackerType, graphPath, effectPath, testState, testDefenses)
    possibleExploit, blockedExploit = graph.checkPossibleTransition(testState)
    graph.getFeasibleStates()
