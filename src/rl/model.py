import torch
import torch.nn as nn
import torch.nn.functional as F 

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()

    return F.softmax(vector, dim=dim)
class Model(nn.Module):
    def __init__(self, dim_observation, n_actions):
        super(Model, self).__init__()
        
        self.n_actions = n_actions
        self.dim_observation = dim_observation
        
        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            nn.Softmax(dim=0)
        )
        
    def forward(self, state):
        return self.net(state)
    
    def select_action(self, state):
        action = torch.multinomial(self.forward(state), 1)
        return action

class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=-1)
        return out

class ValueNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ActorNetwork2(nn.Module):

    def __init__(self, input_size, alert_size, hidden_size):
        super(ActorNetwork2, self).__init__()
        self.input_size = input_size - 1
        self.alert_size = alert_size
        self.embedding = nn.Embedding(2**alert_size * 2, hidden_size)
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc4 = nn.Linear(2*hidden_size, self.alert_size)

    def forward(self, x):
        belief = x[:,:-1]
        alert= x[:,-1].int() 
        bin_val = binary(alert, self.alert_size)
        bin_val[:,-1] = ~(bin_val.sum(-1)>0)
        out = F.relu(torch.concat([self.fc1(belief),self.embedding(alert)], dim=-1))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.softmax(self.fc4(out), dim=-1)
        # out = masked_log_softmax(self.fc3(out), bin_val, dim = -1)
        # out = F.softmax(self.fc3(out) + mask, dim=-1)
        return out

class ValueNetwork2(nn.Module):

    def __init__(self, input_size, alert_size, hidden_size, output_size):
        super(ValueNetwork2, self).__init__()
        self.input_size = input_size - 1
        self.alert_size = alert_size
        self.embedding = nn.Embedding(2**alert_size, hidden_size)
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc4 = nn.Linear(2*hidden_size, output_size)

    def forward(self, x):
        belief = x[:,:-1]
        alert= x[:,-1].int()
        out = F.relu(torch.concat([self.fc1(belief),self.embedding(alert)], dim=-1))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.fc4(out)
        return out

class SAModel(nn.Module):
    def __init__(self, dim_belief, dim_alert):
        super(Model, self).__init__()
        
        self.dim_belief = dim_belief
        self.dim_alert = dim_alert

        self.embedding = nn.Embedding(2**self.dim_alert, 16)
        self.fc1 = nn.Linear(in_features=self.dim_belief, out_features=16)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=32, out_features=self.dim_alert)
        self.softmax = nn.Softmax()
        
    def forward(self, belief, alert):
        belief = belief.reshape(-1)
        alert = self.embedding(alert)
        fc1_out = self.fc1(belief)
        fc1_cat = torch.concat(fc1_out, self.embedding(alert), dim=1)
        fc1_act = self.activation1(fc1_cat)

        fc2_out = self.fc2(fc1_act)
        fc2_act = self.activation2(fc2_out)
        
        fc3_out = self.fc3(fc2_act)
        output  = self.softmax(fc3_out)
        return output
    
    def select_action(self, belief, alert):
        action = torch.multinomial(self.forward(belief, alert), 1)
        return action