# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:37:41 2020

@author: Xinyu HAN
"""

import pandas as pd
import numpy as np
import argparse
import time
import copy
import networkx as nx
from model import *
from tools import *
import os
import copy
import datetime
import scipy.io as scio  


print("Data Loading...")
Lyapunov_E={'Lorenz':0.895,'Rossler':0.071,'Henon':0.419,'M_G':1,'hadcet':1}
Data_Length={'Lorenz':20000,'Rossler':65000,'Henon':20000,'M_G':20000,'hadcet':4343}
L_train_0={'Lorenz':16000,'Rossler':30000,'Henon':16000,'M_G':12000,'hadcet':4224}
L_test_0={'Lorenz':4000,'Rossler':35000,'Henon':100,'M_G':300,'hadcet':119}
transient_0={'Lorenz':1000,'Rossler':1000,'Henon':1000,'M_G':1000,'hadcet':600}
DT={'Lorenz':0.01,'Rossler':0.01,'Henon':1,'M_G':1,'hadcet':1}

#data_name='hadcet'
#data_initial=((pd.read_csv('./datasets/hadcet.csv')).values)[:,1].reshape(-1,1)
#L_train=L_train_0[data_name]
#L_test=L_test_0[data_name]
#
#train_data=copy.deepcopy(data_initial[:L_train,:])
#test_data=copy.deepcopy(data_initial[L_train:(L_train+L_test),:])

data_name='Lorenz'
data_initial=((pd.read_csv('./datasets/Lorenz.csv')).values)[:,1:4]
L_train=L_train_0[data_name]
L_test=L_test_0[data_name]

train_data=copy.deepcopy(data_initial[:L_train,:])
test_data=copy.deepcopy(data_initial[L_train:(L_train+L_test),:])


# basic RC
rng=np.random.RandomState(42)


N=data_initial.shape[1]
Dr=300
density=0.01
rho=0.4
delta=0.01
b=0.01
transient=transient_0[data_name]
# input-to-reservoir matrix
W_in=rng.rand(Dr, N)*2-1


network_name=["ER",'DCG','DAG']
index_R=0
Network_weight=rng.rand(Dr,Dr) 

R_network_0=(Network_initial(network_name[index_R],network_size=Dr,density=density,Depth=z+1,MC_configure=MC_configure)).T
R_network_0=np.triu(np.multiply(R_network_0,Network_weight))+np.triu(np.multiply(R_network_0,Network_weight)).T
R_network=R_network_0/np.max(np.abs(np.linalg.eigvals(R_network_0)))

RC=reservoir_computing(
          N = N,
          Dr =Dr,
          rho=rho,
          delta =delta,
          b=b,
          transient= transient,
          R_network=R_network,
          W_in=W_in)

Train_output=RC.Training_phase(train_data[:-1,:],train_data[1:,:],index_method=0)
            
Pred_test=RC.Predicting_phase(Pre_L=L_test)
 
plot_figure(test_data,Pred_test,number=L_test,Lt=1.0/Lyapunov_E[data_name],dt=DT[data_name],index=1)   
    
    

    
# Interpretable EC

rng=np.random.RandomState(42)

N=data_initial.shape[1]
Dr=300
density=0.1
rho=1
delta=1.5
b=0
transient=transient_0[data_name]
# input-to-reservoir matrix
W_in=rng.rand(Dr, N)*2-1

initial_matrix=np.dot(W_in,train_data[transient:,:].T)
result_max=np.max(initial_matrix)
result_min=np.min(initial_matrix)
intial_expand=0.5*(result_max+result_min)
initial_radius=0.5*(result_max-result_min)
b=0-delta/initial_radius*intial_expand
delta=delta/initial_radius

network_name=["ER",'DCG','DAG']
index_R=2
Network_weight=rng.rand(Dr,Dr) 
MC_configure={}
MC_configure['number']=np.array([180,30,30,30,30])
MC_configure[1]=np.array([1])
MC_configure[2]=np.array([1,2])
MC_configure[3]=np.array([1,2,3])
MC_configure[4]=np.array([1,2,3,4])
MC_configure[5]=np.array([1,2,3,4,5])

R_network_0=(Network_initial(network_name[index_R],network_size=Dr,density=density,Depth=z+1,MC_configure=MC_configure)).T
R_network_1=np.multiply(R_network_0,Network_weight)
W_sum=R_network_1.sum(axis=1,keepdims=1)
W_sum[W_sum==0]=1
R_network=R_network_1*1.0/W_sum
        

#all node is source node
rg=nx.from_numpy_matrix(((R_network>0)*1).T,create_using=nx.DiGraph())
print(pd.Series(node_cluster1(rg)).value_counts())

RC=reservoir_computing(
          N = N,
          Dr =Dr,
          rho=rho,
          delta =delta,
          b=b,
          transient= transient,
          R_network=R_network,
          W_in=W_in)

Train_output=RC.Training_phase(train_data[:-1,:],train_data[1:,:],index_method=4)
            
Pred_test=RC.Predicting_phase(Pre_L=L_test)
 
plot_figure(test_data,Pred_test,number=L_test,Lt=1.0/Lyapunov_E[data_name],dt=DT[data_name],index=1)       


#memory capacity 
rng=np.random.RandomState(42)

N=1
data_initial=(rng.rand(20000,N))
tuo=100
MC=np.zeros((tuo))
i=tuo+1
train_data=data_initial[i:(16000+i),:]
test_data=data_initial[(16000+i):,:]

N=data_initial.shape[1]
Dr=60
density=0.1
rho=1
delta=0.01
b=0
transient=1000
# input-to-reservoir matrix
W_in=rng.rand(Dr, N)*2-1

initial_matrix=np.dot(W_in,train_data[transient:,:].T)
result_max=np.max(initial_matrix)
result_min=np.min(initial_matrix)
intial_expand=0.5*(result_max+result_min)
initial_radius=0.5*(result_max-result_min)
b=0-delta/initial_radius*intial_expand
delta=delta/initial_radius

network_name=["ER",'DAG']
index_R=1
Network_weight=rng.rand(Dr,Dr) 
MC_configure={}
MC_configure['number']=np.array([10]*6)
MC_configure[1]=np.array([1])
MC_configure[2]=np.array([2])
MC_configure[3]=np.array([3])
MC_configure[4]=np.array([4])
MC_configure[5]=np.array([5])
MC_configure[6]=np.array([6])   
R_network_0=(Network_initial(network_name[index_R],network_size=Dr,density=density,Depth=6,MC_configure=MC_configure)).T*(Network_weight+Network_weight.T)/2
W_sum=np.sum(R_network_0,0)
W_sum[W_sum==0]=1
R_network=R_network_0*1.0/W_sum

#top 10 nodes are the source nodes
W_in[MC_configure['number'][0]:,:]=0

souce_node_index=np.ones(Dr)
souce_node_index[MC_configure['number'][0]:]=0

rg=nx.from_numpy_matrix(((R_network>0)*1).T,create_using=nx.DiGraph())
print(pd.Series(node_cluster1(rg,souce_node_index)).value_counts())

RC=reservoir_computing(
          N = N,
          Dr =Dr,
          rho=rho,
          delta =delta,
          b=b,
          transient= transient,
          R_network=R_network,
          W_in=W_in)

MC_C=np.zeros((tuo))
for i in range(tuo,0,-1):
    Expect_output=data_initial[i:(16000+i),:]
    pred_train=RC.Training_phase(train_data,Expect_output,index_method=4)
    MC_C[tuo-i]=memory_capacity(pred_train ,Expect_output,length=100)
    print(MC_C[tuo-i])
plt.plot(MC_C)
print(np.sum(MC_C[MC_C>0.1]))