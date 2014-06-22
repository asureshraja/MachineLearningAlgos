'''
Logistic Regression
@author: suresh
'''

import numpy as np
from math import e

class LogisticRegression():
    def __init__(self,x,y):
        self.x=x
        self.x=np.insert(self.x,0,np.ones(self.x.shape[0]),1)
        self.y=y
        self.theta=np.zeros(shape=(self.x.shape[1]))
        
    def h(self,x):
        return 1/(1+e**(-np.array([x]).dot(self.theta)))[0]
    
    def Train(self,iters=1000,learnrate=0.1,l1reg=False,l2reg=False,lamda=0.01):
        if l1reg==True:
            for ite in xrange(0,iters):
                for i in xrange(0,self.x.shape[0]):
                    for j in xrange(1,self.x.shape[1]):
                        pred=self.h(self.x[i])
                        delta=self.x[i,j]*(pred-self.y[i][0])
                        if self.theta[j]==0:
                            temp=0
                        elif self.theta[j]>0:
                            temp=1
                        else:
                            temp=-1

                        self.theta[j]=self.theta[j] - ((learnrate* delta) + lamda * temp)
                    #print self.theta
        elif l2reg==True:
            for ite in xrange(0,iters):
                for i in xrange(0,self.x.shape[0]):
                    for j in xrange(0,self.x.shape[1]):
                        pred=self.h(self.x[i])
                        delta=self.x[i,j]*(pred-self.y[i][0])
                        self.theta[j]=self.theta[j] - (learnrate* delta) + lamda * self.theta[j]
                    #print self.theta
        else:
            for ite in xrange(0,iters):
                for i in xrange(0,self.x.shape[0]):
                    for j in xrange(0,self.x.shape[1]):
                        pred=self.h(self.x[i])
                        delta=self.x[i,j]*(pred-self.y[i][0])
                        self.theta[j]=self.theta[j] - (learnrate* delta)
                    #print self.theta
        return self.theta

    def Predict(self,x=None,y=None):

        return 1/(1+e**(-np.array(self.x).dot(self.theta)))


datasetx=np.array([[1,1,1,0],[1,1,1,1],[1,1,0,1],[1,1,0,0]])
datasety=np.array([[1],[1],[1],[0]])

lg=LogisticRegression(datasetx,datasety)
print lg.Train(l2reg=True)
print lg.Predict()