import json
import numpy as np
from src.util.auxiliaries import checkAttackerType, convKeyInt
EPS = np.finfo(float).eps

class IDS:
    def __init__(self, attackerType, idsPath, fp_rate, seed):
        self.attackerType = attackerType
        with open(idsPath, 'r') as f:
            idsRate = json.load(f)
        idsTPMatrix = []
        self.type2int = idsRate["AttackerType"]
        int2type = {v:k for k,v in self.type2int.items()}
        for k in sorted(int2type.keys()):
            idsTPMatrix.append([v for k,v in idsRate["TP"][int2type[k]].items()])
        self.idsTPMatrix = np.transpose(np.array(idsTPMatrix), (1,2,0))
        idsFPMatrix = []
        for k in sorted(int2type.keys()):
            idsFPMatrix.append(idsRate["FP"][int2type[k]])
        self.idsFPMatrix = np.array(idsFPMatrix)
        if fp_rate is not None:
            self.idsFPMatrix = np.array([fp_rate] * self.idsFPMatrix.shape[0])
        self.idsTPRate = self.idsTPMatrix[:,:,idsRate["AttackerType"][attackerType]]           
        self.idsFPRate = self.idsFPMatrix[idsRate["AttackerType"][attackerType]] 

        # self.idsTPRate[self.idsTPRate<EPS] = EPS
        # self.idsFPRate[self.idsFPRate<EPS] = EPS

        # self.idsTPRate[self.idsTPRate>1-EPS] = 1-EPS
        # self.idsFPRate[self.idsFPRate>1-EPS] = 1-EPS

        self.allAlerts = np.array(range(len(self.idsTPRate)))
        self.numAlert = self.idsTPRate.shape[0]
        self.random_state = np.random.RandomState(seed)
    
    def genAlerts(self, chosenAttack):
        alerts = []
        probNoAlert = (1 - self.idsFPRate) * np.prod(1- self.idsTPRate[:,chosenAttack], axis=1)
        probAlert = 1 - probNoAlert 
        alerts = self.allAlerts[self.random_state.rand(*self.allAlerts.shape) < probAlert]
        return alerts
        
class MyIDS(IDS):
    def __init__(self, attackerType, idsPath, fp_rate, seed):
        super().__init__(attackerType, idsPath, fp_rate, seed)
    
    def genAlerts(self, E_t):
        y_t = []
        probTP = self.idsTPRate[:,E_t]
        probFP = self.idsFPRate
        probAlert = np.concatenate([probTP, np.expand_dims(probFP, axis=-1)], axis= 1)
        triggeredAlerts = self.random_state.rand(*probAlert.shape) < probAlert
        y_t_true = np.arange(self.numAlert)[np.any(triggeredAlerts[:,:-1], axis=-1)]
        y_t_false = np.arange(self.numAlert)[triggeredAlerts[:,-1]]
        y_t = np.arange(self.numAlert)[np.any(triggeredAlerts, axis = 1)]

        return y_t, y_t_true, y_t_false