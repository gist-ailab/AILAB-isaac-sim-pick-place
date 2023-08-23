import torch.nn as nn
import math
import torch

### model that only with channel 2 hiddenlayer that all of layer can be visualize

class FCN(nn.Module):
    def __init__(self, num_layer, act_func) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.feature = []
        self.layer_list = nn.Sequential( *([nn.Linear(2,2)] * num_layer))
        self.last_layer = nn.Linear(2,1)
        
        ### choose activation function
        if act_func == "ReLU":
            self.act = nn.ReLU()
        elif act_func == "LeakyReLU":
            self.act = nn.LeakyReLU(-1.0)
        else:
            self.act = None
        for i in range(num_layer):
            self.layer_list[i].weight = torch.nn.Parameter(torch.tensor([[1.0,0.0],[0.0,1.0]]))
        
    def forward(self, x):
        self.reset()
        self.feature.append(x.detach().numpy())
        for i in range(self.num_layer):
            x = self.layer_list[i](x)
            if self.act != None:
                x = self.act(x)
            np_x = x.detach().numpy()
            self.feature.append(np_x)
        x = self.last_layer(x)
        self.feature.append(x.detach().numpy())
        return x, self.feature

    def reset(self):
        self.feature = []
        
    def get_layer_list(self):
        layer_list = []
        layer_list.append(self.layer_list)
        layer_list.append(self.last_layer)
        return layer_list
    
### model with larger channel that can train well only visualize last and pernultimate layer

class FCN_exp2(nn.Module):
    def __init__(self, num_layer, act_func) -> None: 
        super().__init__() 
        self.num_layer = num_layer          
        self.layer_list = nn.Sequential() 
        for i in range(1,num_layer+1): 
            self.layer_list.add_module("layer_"+str(i), nn.Linear(int(math.pow(2,i)),int(math.pow(2,i+1)))) 
            
            ### choose activate function
            if act_func == "ReLU":
                self.layer_list.add_module("ReLU_"+str(i), nn.ReLU())
            elif act_func == "LeakyReLU":
                self.layer_list.add_module("LeakyReLU_"+str(i), nn.LeakyReLU(-1.0)) 
        
        self.visual_layer = nn.Linear(int(math.pow(2,num_layer+1)), 2) 
        self.last_layer = nn.Linear(2,1)  
    
    def forward(self, x):
        self.reset()
        self.feature.append(x.detach().numpy()) 
        x = self.layer_list(x) 
        latent = self.visual_layer(x) 
        self.feature.append(latent.detach().numpy())
        output = self.last_layer(latent)
        self.feature.append(output.detach().numpy()) 
        return output, self.feature
    
    def reset(self):
        self.feature = []
        
    def get_layer_list(self):
        layer_list = []
        layer_list.append(self.layer_list)
        layer_list.append(self.visual_layer)
        layer_list.append(self.last_layer)
        return layer_list
        
def mm(model, feature_list, num_layer):
    for i in range(num_layer):
        if feature_list[i+1][0][0] == 0:
            feature_list[i+1][0][0] = 1e-4
        if feature_list[i][0][1] ==0:
            feature_list[i][0][1] = 1e-4
        x = math.atan(feature_list[i+1][0][1]/feature_list[i+1][0][0])-math.atan(feature_list[i][0][1]/feature_list[i][0][0],2)
        model.get_layer_list()[0][i].weight = torch.nn.Parameter(torch.tensor([[math.cos(x),-math.sin(x)],[math.sin(x),math.cos(x)]]))
    
    
if __name__ == "__main__":
    model = FCN(3, "ReLU")
    for i in range(len(model.modules)):
        model.modules[i]