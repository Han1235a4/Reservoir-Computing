# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:22:03 2020

@author: Xinyu Han
"""

import numpy as np
import pandas as pd
import pickle
from itertools import combinations, permutations
import random
import copy
from matplotlib import pyplot as plt
import networkx as nx
import os  
from sklearn import linear_model
from scipy.linalg import orth
from sklearn.decomposition import PCA
import math
from sklearn.cluster import KMeans,Birch
from sklearn.manifold import TSNE
from scipy.integrate import odeint
import matplotlib.colors as colors
from itertools import cycle
#from utils import tsne
from networkx.algorithms import bipartite
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.linear_model import LassoCV,RidgeCV,Ridge,ElasticNetCV,orthogonal_mp,OrthogonalMatchingPursuit
from sklearn.decomposition import PCA
from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
from sklearn.utils.extmath import safe_sparse_dot, row_norms
from scipy import special, stats
from scipy.sparse import issparse
from sys import path

rng=np.random.RandomState(42)
# calculate the memory capacity ot RC model
def memory_capacity(Prediction=None, Output=None, length=1000):
    if Prediction.shape[1]==1:
        sumR_coor=(np.corrcoef(Prediction[-length:].T, Output[-length:].T)[0,1])**2
    else:
        sumR_coor=0
        for i in range(Prediction.shape[1]):
            sumR_coor=sumR_coor+(np.corrcoef(Prediction[-length:,i], Output[-length:,i])[0,1])**2
        sumR_coor=sumR_coor*1.0/Prediction.shape[1]
    return sumR_coor

def R_shuffle(node_number=0,path_length=0):
    x = [np.random.random() for i in range(path_length)]
    e = [int(i / sum(x) * (node_number-path_length)) + 1 for i in x] 
    re = node_number - sum(e)
    u = [np.random.randint(0, path_length- 1) for i in range(re)] 
    for i in range(re):
        e[u[i]] += 1
    return e

def Network_initial(network_name=None,network_size=300,density=0.2,Depth=10,MC_configure=None):

    if network_name is "ER":
        rg=nx.erdos_renyi_graph(network_size,density,directed=False)#ER
        R_initial=nx.adjacency_matrix(rg).toarray()
    elif network_name is "DCG":
        rg=nx.erdos_renyi_graph(network_size,density,directed=True)#ER 
        R_initial=nx.adjacency_matrix(rg).toarray()    
    elif network_name is "DAG":
        if MC_configure is not None:
            xx=np.append(0,np.cumsum(MC_configure['number']))
            for i in range(xx.shape[0]-1):
                Reject_index=1
                for j in range(0,xx.shape[0]-1):
                    if len(MC_configure[i+1])==np.sum(np.isin(MC_configure[i+1],MC_configure[j+1]+1)):
                        Reject_index=0
                if Reject_index==1 and (MC_configure[i+1]!=1).all():
                    print("fail to construct the DAN under current Memory commnity strcutrue configuration")                    
                    Reject_index=2
            if Reject_index !=2:
                R_initial_0=np.zeros((network_size,network_size))
                for i in range(xx.shape[0]-1):
                    for j in range(xx.shape[0]-1):
                        if len(MC_configure[i+1])==np.sum(np.isin(MC_configure[i+1]+1,MC_configure[j+1])):
                            R_initial_0[xx[i]:xx[i+1],xx[j]:xx[j+1]]=1
                R_initial= np.triu(R_initial_0,1)
            else:
                R_initial=None
            
        else:
            xx=R_shuffle(network_size,Depth)
            # xx=np.array([3,4,3])
            # xx=np.array([60,60,60,60,60])
            # xx=np.array([30,30,30,30,30,30,30,30,30,30])*3
            rg = nx.complete_multipartite_graph(*tuple(xx))  
            x=nx.adjacency_matrix(rg).toarray()
            R_initial= np.triu(x,1)  
        # R_initial= np.tril(x,1)  
        Real_density=np.sum(R_initial>0)*1.0/(network_size**2)
        if Real_density>0 and density<Real_density:
            R_initial[rng.rand(*R_initial.shape) <= (1.0-density/Real_density)] = 0 
        R_initial= np.triu(R_initial,1)  
    return R_initial

# the method of training the readout matrix
def traing_Wout(train_data,R_state,index=0,k=0.8):
    W_out= np.zeros((R_state.shape[1], train_data.shape[1]))
    if index==0:
        W_out=np.dot(np.linalg.pinv(R_state),train_data)#Dr*N
    else:         
        alphas = 10**np.linspace(-4,2,7)
        if index==1:
            base_cv=LassoCV(alphas = alphas,fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        if index==2:
            base_cv = RidgeCV(alphas = alphas,fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        if index==3:
            base_cv=ElasticNetCV(alphas = alphas,fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        
        if index==4:
            anova_filter = SelectKBest(f_regression, k=int(R_state.shape[1]*k))#int(self.n_reservoir*0.8 ))#k_number)
            
        if index==5:
            base=linear_model.LinearRegression(fit_intercept=True)
            anova_filter = RFECV(base)
        if index==6:
            base_cv=OrthogonalMatchingPursuit(fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        
        clf = Pipeline([
         ('feature_selection', anova_filter),
         ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
         ])
        for X_i in range(train_data.shape[1]):
            clf.fit(R_state,train_data[:, X_i])
            W_out[clf.named_steps['feature_selection'].get_support(),X_i]=clf.named_steps['Linearregression'].coef_
    return W_out

# the valid time     
def plot_figure(test_output,pred_test,number,Lt=1,dt=0.1,name=None,index=1):
    '''
    index 1  plot the figure
    Lt the max predit time
    Et the judege indicator
    Etstander the stander which we can accept
    '''
    Et=np.linalg.norm(pred_test[:number]-test_output[:number],ord=2,axis=1)/np.linalg.norm(test_output[:number],ord=2,axis=1)
    t = np.arange(0, test_output[:number].shape[0]*dt,dt)
    plt.show()
    t = np.arange(0, test_output.shape[0]*dt,dt)
    t=t/Lt    
    t=t[:number]
    Etstander=0.5#the given error criterion 
    if np.where(Et>=Etstander)[0].shape[0]>0:
        Etindex=np.min(np.where(Et>=Etstander))
    else:
        print("----------")
        Etindex=-1   
    if index==1:
        if test_output.shape[1]==1:
            ax = plt.plot()
            plt.plot(t,test_output[:number,0],color='blue',label='Actual')
            plt.plot(t,pred_test[:number,0],color='red',linestyle="dashed" ,label='Predict')
            plt.legend(fontsize='x-small',bbox_to_anchor=(1.05,1))
            plt.ylabel("x(t)")
            if name is not None:
                plt.savefig(name,format='pdf',bbox_inches='tight')
            plt.show()
        if test_output.shape[1]==2:
            ax1 = plt.subplot(211)
            plt.plot(t,test_output[:number,0],color='blue',label='Actual')
            plt.plot(t,pred_test[:number,0],color='red',linestyle="dashed",label='Predict')
            plt.legend(fontsize='x-small',bbox_to_anchor=(1.05,1))
            plt.ylabel("x(t)")
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            
            # share x only
            ax2 = plt.subplot(212, sharex=ax1)
            plt.plot(t,test_output[:number,1],color='blue')
            plt.plot(t,pred_test[:number,1],color='red',linestyle="dashed")
            plt.ylabel("y(t)")
            # make these tick labels invisible
            
            plt.xlim(0,round(t[-1]))
            plt.xlabel("$\lambda_{max}t$")
            #plt.savefig(name,format='pdf',bbox_inches='tight')
            if name is not None:
                plt.savefig(name,format='pdf',bbox_inches='tight')
            plt.show()
        if test_output.shape[1]>=3:
            ax1 = plt.subplot(311)
            plt.plot(t,test_output[:number,0],color='blue',label='Actual')
            plt.plot(t,pred_test[:number,0],color='red',linestyle="dashed",label='Predict')
            plt.legend(fontsize='x-small',bbox_to_anchor=(1.05,1))
            plt.ylabel("x(t)")
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            
            # share x only
            ax2 = plt.subplot(312, sharex=ax1)
            plt.plot(t,test_output[:number,1],color='blue')
            plt.plot(t,pred_test[:number,1],color='red',linestyle="dashed")
            plt.ylabel("y(t)")
            # make these tick labels invisible
            plt.setp(ax2.get_xticklabels(), visible=False)
            
            # share x and y
            ax3 = plt.subplot(313, sharex=ax1)
            plt.plot(t,test_output[:number,2],color='blue')
            plt.plot(t,pred_test[:number,2],color='red',linestyle="dashed") 
            plt.ylabel("z(t)")
            plt.xlim(0,round(t[-1]))
            plt.xlabel("$\lambda_{max}t$")
            #plt.savefig(name,format='pdf',bbox_inches='tight')
            if name is not None:
                plt.savefig(name,format='pdf',bbox_inches='tight')
            plt.show()
#    print(t[Etindex])
    return t[Etindex] 

# directed network embedding method  Usage: print(pd.Series(node_cluster1(rg)).value_counts())
def node_cluster1(rg=None,souce_node_index=None): #rg is the Directed acyclic network
    x=nx.adjacency_matrix(rg).toarray()
    DAG_index=nx.is_directed_acyclic_graph(rg)
    din=dict(rg.in_degree()).values()
    din_number=np.array(din)    
    if DAG_index==True:
        node_list=list(rg.nodes()) 
        DAG_len=nx.dag_longest_path_length(rg)
        node_cluster=['1']*np.size(nx.nodes(rg))
        node_index=np.zeros(len(node_list))
        if souce_node_index is not None:
            node_index=souce_node_index
            for i in range(np.size(nx.nodes(rg))):
                if node_index[i]==0:
                    node_cluster[i]=''                    
        else:
            node_index[din_number==0]=1
        for i in range(DAG_len):
            tem_index=np.where(np.sum(x[node_index==1,:],axis=0)!=0)[0].tolist()
            for j in tem_index:
                node_cluster[j]=str(i+2)+node_cluster[j]
            node_index=np.zeros(np.size(nx.nodes(rg)))
            node_index[tem_index]=1
    else:
           
        node_cluster=['1']*np.size(nx.nodes(rg))
        node_index=np.zeros(np.size(nx.nodes(rg)))
        node_index[din_number==0]=1
        cyclic_list=list(nx.simple_cycles(rg))
        cyclic_index1=np.zeros((len(cyclic_list),np.size(nx.nodes(rg))))
        cyclic_index=np.zeros((len(cyclic_list),np.size(nx.nodes(rg))))
        for i in range(len(cyclic_list)):
            for j in range(len(cyclic_list[i])):
                cyclic_index1[i,cyclic_list[i][j]]=1
                cyclic_index[i,cyclic_list[i][j]]=1
                cyclic_index[i,list(nx.descendants(rg,cyclic_list[i][j]))]=1  
        cyclic_index1=cyclic_index1[np.argsort(-np.sum(cyclic_index,axis=1)),:]
        cyclic_index=cyclic_index[np.argsort(-np.sum(cyclic_index,axis=1)),:]   
        node_list=list(rg.nodes())       
        i=1
        while node_list!=[]:
            print(node_list)
            x=nx.adjacency_matrix(rg).toarray()
            din=dict(rg.in_degree()).values()
            din_number=np.array(din)
            node_index=np.zeros(len(node_list))
            node_index[din_number==0]=1
            tem_index=np.where(np.sum(x[node_index==1,:],axis=0)!=0)[0].tolist()
            if np.sum(node_index)>0 and len(tem_index)==0:            
                for j in np.where(din_number==0)[0].tolist():
                    node_cluster[node_list[j]]=str(i+1)+node_cluster[node_list[j]]
                i=i-1
            if np.sum(node_index)==0 and len(tem_index)==0: 
                rg.remove_nodes_from(node_list)
                for j in node_list:
                    node_cluster[j]='inf'+str(i+1)+node_cluster[j]
            if  len(tem_index)>0:                       
                for j in tem_index:
                    tem_index1=np.where(cyclic_index1[:,node_list[j]]==1)[0].tolist()
                    if len(tem_index1)>0:
                        tem_index3=np.where(np.sum(cyclic_index[tem_index1,:],axis=0)>=1)[0].tolist()
                        rg.remove_nodes_from(tem_index3)
                        for z in tem_index3:
                            node_cluster[z]='inf--'+str(i+1)+node_cluster[z]
                            if z in [node_list[zz] for zz in tem_index]:
                                del tem_index[[node_list[zz] for zz in tem_index].index(z)]                     
                    else:
                       node_cluster[node_list[j]]=str(i+1)+node_cluster[node_list[j]]
            rg.remove_nodes_from([node_list[zz] for zz in np.where(din_number==0)[0].tolist()])
            node_list=list(rg.nodes())  
            i=i+1
    return(node_cluster)