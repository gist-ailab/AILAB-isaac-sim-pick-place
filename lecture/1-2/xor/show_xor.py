import torch
import torch.nn as nn
import sys
from visualize import *
from generate_array import get_grid
from vis_dataset import *
from model import FCN
from get_train import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

### get mouse input with pygame 
init_data_array = get_grid()

### preprocess dataset and define loader
train_dataset = VIS_DATASET(init_data_array)
all_dot = DOT_DATASET(len(init_data_array))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
all_loader = torch.utils.data.DataLoader(all_dot, batch_size = 1, shuffle= False)

### define model and hyperparameter can change model's layer 
num_layer = 1
act_function = "LeakyReLU"       ## or "ReLU"

model = FCN(num_layer, act_function)
print(model)

learning_rate = 1e-1
loss_function = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.5)

epoch = 100

### visualize with PyQt5 start
lat = [[[train_dataset[i][0].detach().numpy()]] for i in range(len(train_dataset))]
lab = [train_dataset[i][1].detach().numpy() for i in range(len(train_dataset))]

qApp = QApplication(sys.argv)
aw = VisualizeAllLayer(lat, lab, num_layer)

### train and visualize
for t in range(epoch+1):
    loss = train(train_loader, model, loss_function, optimizer, scheduler, num_layer)
    threshold = get_threshold(train_loader, model)
    dot_output, dot_feature, train_out, train_feature, train_label = test(all_loader, train_loader, model)
    aw.update_var(train_feature, dot_feature, train_label, t, loss, threshold)
    
sys.exit(qApp.exec_())