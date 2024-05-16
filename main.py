
import torch
##read json
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset,ConcatDataset
import itertools
import string
from tqdm import tqdm
from mlp import MLP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from mlp import gradients
epochs=10
h=1000
N=1000
N1=100
N2=1000
def setup_seed(seed):
    torch.manual_seed(seed)
class lid_driven:
    def __init__(self,nx,ny,Re,dt,timesteps):
        self.nx=nx
        self.x_min=0
        self.x_max=1
        self.ny=ny
        self.y_min=0
        self.y_max=1
        self.Re=Re
        self.dt=dt
        self.timesteps=timesteps
        self.dx=1/nx
        self.dy=1/ny
        self.data_size=nx*ny
        self.model=MLP(3,100,100,3,nn.ReLU())
        self.criteria=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.01)
        self.generate_up_boundary()
        self.generate_other_boundary()

        self.generate_c()

    def generate_up_boundary(self):
        x=torch.linspace(self.x_min,self.x_max,self.data_size)
        y=torch.ones(self.data_size)*self.y_max
        self.up_boundary_data_set=TensorDataset(torch.stack((x,y),dim=1))       
     
        self.up_boundary_data_loader=DataLoader(dataset=self.up_boundary_data_set,batch_size=10,shuffle=True,generator=torch.Generator(device='cuda'))
        self.up_boundary_data_loader=itertools.cycle(self.up_boundary_data_loader)
    def generate_other_boundary(self):
        data_size=self.data_size
        ##generate left
        x=torch.ones(data_size)*self.x_min
        y=torch.linspace(self.y_min,self.y_max,data_size)
        left=torch.stack((x,y),dim=1)
        
        ##generate right
        x=torch.ones(data_size)*self.x_max
        y=torch.linspace(self.y_min,self.y_max,data_size)
        right=torch.stack((x,y),dim=1)

        #generate bottom
        x=torch.linspace(self.x_min,self.x_max,data_size)
        y=torch.ones(data_size)*self.y_min
        bottom=torch.stack((x,y),dim=1)
        self.other_boundary_data_set=TensorDataset(torch.cat((left,right,bottom),dim=0))
        self.other_boundary_data_loader=DataLoader(dataset=self.other_boundary_data_set,batch_size=100,shuffle=True,generator=torch.Generator(device='cuda'))
        self.other_boundary_data_loader=itertools.cycle(self.other_boundary_data_loader)
    
    def generate_c(self):
        x=torch.linspace(self.x_min,self.x_max,self.data_size)
        y=torch.linspace(self.y_min,self.y_max,self.data_size)
        self.c_data_set=TensorDataset(torch.stack((x,y),dim=1))
        self.c_data_loader=DataLoader(dataset=self.c_data_set,batch_size=500,shuffle=True,generator=torch.Generator(device='cuda'))
    def pde_loss(self,x,y,t):
        u,v,p=torch.split(self.model(torch.cat((x,y,t),dim=1)),split_size_or_sections=1,dim=1)
        input=torch.cat((x,y,t),dim=1)
        u_t=gradients(u,t)
        v_t=gradients(v,t)
        p_x=gradients(p,x)
        p_y=gradients(p,y)
        u_x=gradients(u,x)
        u_y=gradients(u,y)
        v_x=gradients(v,x)
        v_y=gradients(v,y)

        u_xx=gradients(u,x,2)
        u_yy=gradients(u,y,2)
        v_xx=gradients(v,x,2)
        v_yy=gradients(v,y,2)
        Re=self.Re
        inv_Re=1.0/Re
        Loss_x=self.criteria(u_t-inv_Re*(u_xx+u_yy)+u*u_x+v*u_y+p_x,torch.zeros_like(u_t))
        Loss_y=self.criteria(v_t-inv_Re*(v_xx+v_yy)+u*v_x+v*v_y+p_y,torch.zeros_like(v_t))
        div_loss=self.criteria(u_x+v_y,torch.zeros_like(u_x))
        total_loss=Loss_x+Loss_y+div_loss
        return torch.tensor(total_loss)
    

    def up_boundary_loss(self,x,y,t):
        u,v,p=torch.split(self.model(torch.cat((x,y,t),dim=1)),split_size_or_sections=1,dim=1)
        loss_u=self.criteria(u,torch.ones_like(u))
        loss_v=self.criteria(v,torch.zeros_like(v))
        return loss_u+loss_v
      
    def other_boundary_loss(self,x,y,t):
        u,v,p=torch.split(self.model(torch.cat((x,y,t),dim=1)),split_size_or_sections=1,dim=1)
        loss_u=self.criteria(u,torch.zeros_like(u))
        loss_v=self.criteria(v,torch.zeros_like(v))
        return loss_u+loss_v

    def initial_time_loss(self,x,y,t):
        u,v,p=torch.split(self.model(torch.cat((x,y,t),dim=1)),split_size_or_sections=1,dim=1)
        loss_u=self.criteria(u,torch.zeros_like(u))
        loss_v=self.criteria(v,torch.zeros_like(v))
        loss_p=self.criteria(p,torch.zeros_like(p))

        return loss_u+loss_v+loss_p
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

    def train(self,_t,show=False):
        ##generate boundary
        up_boundary_data=self.up_boundary_data_loader
        other_boundary_data=self.other_boundary_data_loader
        c_data=self.c_data_loader
        for up,other,c in zip(up_boundary_data,other_boundary_data,c_data):
            self.optimizer.zero_grad()
            x_c=c[0][:,0].unsqueeze(1)
            x_c.requires_grad=True
        
            y_c=c[0][:,1].unsqueeze(1)
            y_c.requires_grad=True
        
            t_c=torch.ones_like(x_c)*_t
            t_c.requires_grad=True
            pde_loss=self.pde_loss(x_c,y_c,t_c)

            x_up=up[0][:,0].unsqueeze(1)
            x_up.requires_grad=True
            y_up=up[0][:,1].unsqueeze(1)
            y_up.requires_grad=True
            t_up=torch.ones_like(x_up)*_t
            t_up.requires_grad=True
            up_loss=self.up_boundary_loss(x_up,y_up,t_up)

            x_other=other[0][:,0].unsqueeze(1)
            x_other.requires_grad=True
            y_other=other[0][:,1].unsqueeze(1)
            y_other.requires_grad=True
            t_other=torch.ones_like(x_other)*_t
            t_other.requires_grad=True            
            other_loss=self.other_boundary_loss(x_other,y_other,t_other)

            
            initial_loss=self.initial_time_loss(x_c,y_c,torch.zeros_like(x_c))
            loss=pde_loss+up_loss+other_loss+initial_loss
            loss.backward()
            self.optimizer.step()
            if  show:
                ##save the model
                torch.save(self.model.state_dict(),f"model.pth")
                print(f"loss:{loss}")
                print(f"pde_loss:{pde_loss}")
                print(f"up_loss:{up_loss}")
                print(f"other_loss:{other_loss}")
                print(f"initial_loss:{initial_loss}")
                print(f"t:{_t}")
                print("====================================")


            


def visualization(time,scene,config:string ):
    if config=="velocity":
        scene.load_model(f"model.pth")
        x=torch.linspace(0,1,100)
        y=torch.linspace(0,1,100)
        X,Y=torch.meshgrid(x,y)
        #visualize the velocity
        with torch.inference_mode():
            u,v,p=torch.split(scene.model(torch.cat((X.reshape(-1,1),Y.reshape(-1,1),torch.ones_like(X.reshape(-1,1))*time),dim=1)),split_size_or_sections=1,dim=1)
        u=u.detach().cpu().numpy().reshape(100,100)
        v=v.detach().cpu().numpy().reshape(100,100)
        p=p.detach().cpu().numpy().reshape(100,100)
        norm=u*u+v*v
        #show the norm
        plt.imshow(norm)
        plt.draw()
        #save the figure
        plt.savefig('./figures/figure_{}.jpg'.format(time))



def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    config=config
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.set_default_device('cuda')
    scene=lid_driven(config['nx'],config['ny'],config['Re'],config['dt'],config['timesteps'])
    scene.model.train()
    setup_seed(888888)
    for epoch in range(0,epochs):
        for t in tqdm(range(scene.timesteps)):
            if t%100==0:
                scene.train(t*scene.dt,True)
            else:
                scene.train(t*scene.dt,False)

    
    for t in range(0,scene.timesteps):
        visualization(t,scene,"velocity")



   
    
if __name__=="__main__" :
    main()





