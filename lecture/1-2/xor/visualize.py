from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MyMplCanvas(FigureCanvas):
    def __init__(self, num, parent=None):
        f, axes = plt.subplots(1,num+2)
        self.axes = axes
        self.f = f
        
        for i in range(num+2):
            axes[i].set_facecolor('black')
            axes[i].set_aspect('equal')
            
            
        f.set_size_inches((50,50*(num+2)))
        
        self.compute_initial_figure()
        FigureCanvas.__init__(self, f)
        self.setParent(parent)
    def compute_initial_figure(self):
        pass

class VisualizeAllLayer(QWidget):
    def __init__(self, feature, label, num_layer):
        QMainWindow.__init__(self)
        
        self.num_layer = num_layer
        
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(num = num_layer)
        vbox.addWidget(self.canvas)
        
        self.setLayout(vbox)
        for i in range(len(label)):
            for j in range(len(feature[0])):
                if label[i].item()==0:
                    self.line = self.canvas.axes[0].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='red', s= 100)
                elif label[i].item()==1:
                    self.line = self.canvas.axes[0].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='yellow', s= 100)
        plt.pause(3)

    def update_var(self, feature, dot_feature, label, epoch, loss, threshold):
        for i in range(self.num_layer+2):
            self.canvas.axes[i].cla()
            
        for j in range(len(feature[0])):
            for i in range(len(feature)):
                if j == len(feature[0])-1:
                    self.canvas.axes[j].set_title(f"Epoch: {epoch} Loss : {loss}", fontsize = 20)
                    if label[i].item()==0:
                        self.canvas.axes[j].scatter(feature[i][j][0][0], y = 0, color = 'red', s=100)
                    elif label[i].item()==1:
                        self.canvas.axes[j].scatter(feature[i][j][0][0], y = 0, color = 'yellow', s=100)
                else:
                    if label[i].item()==0:
                        self.canvas.axes[j].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='red', s= 100)
                    elif label[i].item()==1:
                        self.canvas.axes[j].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='yellow', s= 100)
                    self.canvas.axes[j].set_title(f"Layer : {j}", fontsize = 20)
                
            
            for i in range(len(dot_feature)):
                if j == len(dot_feature[0])-1:
                    self.canvas.axes[j].set_title(f"epoch: {epoch} loss : {loss}", fontsize = 20)
                    if dot_feature[i][len(dot_feature[0])-1][0][0]<threshold:
                        self.canvas.axes[j].scatter(dot_feature[i][j][0][0], y = 0, color = 'red', alpha=0.3, s=10)
                    elif dot_feature[i][len(dot_feature[0])-1][0][0]>threshold:
                        self.canvas.axes[j].scatter(dot_feature[i][j][0][0], y = 0, color = 'yellow', alpha=0.3, s=10)
                else:
                    if dot_feature[i][len(dot_feature[0])-1][0][0]<threshold:
                        self.canvas.axes[j].scatter(x = dot_feature[i][j][0][0], y = dot_feature[i][j][0][1], color='red', alpha=0.3, s= 100)
                    elif dot_feature[i][len(dot_feature[0])-1][0][0]>threshold:
                        self.canvas.axes[j].scatter(x = dot_feature[i][j][0][0], y = dot_feature[i][j][0][1], color='yellow', alpha=0.3, s= 100)
                    self.canvas.axes[j].set_title(f"layer : {j}", fontsize = 20)
                    
        plt.pause(0.001)


class LightVisualize(QWidget):
    def __init__(self, feature, label, num_layer):
        QMainWindow.__init__(self)
        
        self.num_layer = num_layer
        
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(num = num_layer)
        vbox.addWidget(self.canvas)
        
        self.setLayout(vbox)
        for i in range(len(label)):
            for j in range(len(feature[0])):
                if label[i].item()==0:
                    self.line = self.canvas.axes[0].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='red', s= 100)
                elif label[i].item()==1:
                    self.line = self.canvas.axes[0].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='yellow', s= 100)
        plt.pause(3)

    def update_var(self, feature, dot_feature, label, epoch, loss, threshold):
        if epoch%10 == 0:
            for i in range(self.num_layer+2):
                self.canvas.axes[i].cla()
                
            for j in range(len(feature[0])):
                for i in range(len(feature)):
                    if j == len(feature[0])-1:
                        self.canvas.axes[j].set_title(f"Epoch: {epoch} Loss : {loss}", fontsize = 20)
                        if label[i].item()==0:
                            self.canvas.axes[j].scatter(feature[i][j][0][0], y = 0, color = 'red', s=100)
                        elif label[i].item()==1:
                            self.canvas.axes[j].scatter(feature[i][j][0][0], y = 0, color = 'yellow', s=100)
                    else:
                        if label[i].item()==0:
                            self.canvas.axes[j].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='red', s= 100)
                        elif label[i].item()==1:
                            self.canvas.axes[j].scatter(x = feature[i][j][0][0], y = feature[i][j][0][1], color='yellow', s= 100)
                        if j == 1:
                            self.canvas.axes[j].set_title("Penultimate Layer", fontsize=20)
                        else:
                            self.canvas.axes[j].set_title(f"Layer : {j}", fontsize = 20)
                
            
                for i in range(len(dot_feature)):
                    if j == 0:
                        if dot_feature[i][len(dot_feature[0])-1][0][0]<threshold:
                            self.canvas.axes[j].scatter(x = dot_feature[i][j][0][0], y = dot_feature[i][j][0][1], color='red', alpha=0.3, s= 100)
                        elif dot_feature[i][len(dot_feature[0])-1][0][0]>threshold:
                            self.canvas.axes[j].scatter(x = dot_feature[i][j][0][0], y = dot_feature[i][j][0][1], color='yellow', alpha=0.3, s= 100)
                        self.canvas.axes[j].set_title(f"layer : {j}", fontsize = 20)
                    
            plt.pause(0.001)