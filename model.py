# -*- coding: utf-8 -*-
"""
Python 2.7
Created on Wed Dec 23 20:36:43 2020

@author: Xinyu HAN
"""

import numpy as np
import pandas as pd
from scipy import special, stats
from tools import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
    
# reservoir computing
class reservoir_computing():
    def __init__(self,N=3,Dr=300, rho=1,delta=0.1,b=0,transient=1000,R_network=None,W_in=None):
        """
        L: 
        N: the input dimension
        Dr: the reservoir network size
        rho: spectral radius of the reservoir network weight matrix
        delta： the scaling parameter of the input-to-reservoir matrix
        b： the bias term
        transient: the number of reservoir state data  to be deleted  
        
        R_network: the reservoir network weight matrix
        W_in : the input-to-reservoir matrix
        """
        self.N=N
        self.Dr=Dr
        self.rho=rho
        self.delta=delta
        self.b=b
        self.transient=transient
        
        self.R_network=R_network
        self.W_in=W_in
        
    
    def Training_phase(self, train_data=None,Train_expect=None,index_method=0):
        L=train_data.shape[0]
        R_state = np.zeros((L, self.Dr))
        W_out= np.zeros((self.Dr, self.N))
        Pre_train_output= np.zeros((L, self.N))
         
        R_state[0,:]=np.tanh(np.dot(self.R_network, R_state[0,:])
                             +self.delta*np.dot(self.W_in,train_data[0,:])+self.b)
        for i in range(1,L):
            R_state[i,:]=np.tanh(np.dot(self.R_network, R_state[i-1,:])
                         +self.delta*np.dot(self.W_in,train_data[i,:])+self.b)
        
        W_out=traing_Wout(Train_expect[self.transient:,:],R_state[self.transient:,:],index=index_method,k=0.8)
#        print(W_out.shape)
        
        Pre_train_output=np.dot(R_state,W_out)
         
        self.W_out=W_out
        self.laststate=R_state[-1,:]
        self.lastinput=Train_expect[-1,:]
         
        return Pre_train_output#,W_out,R_state
    
    def Predicting_phase(self,Pre_L=100):
 
        outputs=np.zeros((Pre_L,self.N))
        R_state=np.zeros((Pre_L,self.Dr))
        
        R_state[0,:]=np.tanh((np.dot(self.R_network,self.laststate)
                             +self.delta*np.dot(self.W_in,self.lastinput)+self.b))
        outputs[0,:]=np.dot(R_state[0, :],self.W_out,)
        for i in range(1,Pre_L):
            R_state[i, :] = np.tanh(np.dot(self.R_network, R_state[i-1,:])
                             +self.delta*np.dot(self.W_in,outputs[i-1,:])+self.b)
            
            outputs[i, :] = np.dot(R_state[i, :],self.W_out)
            
        return outputs
    
