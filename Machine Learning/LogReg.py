import numpy as np
from math import e
#Batch Gradient Descent
class LogisticRegressionSGD():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.theta=np.zeros(shape=(x.shape[1]))
    def h(self,x,theta):
        return e**(np.dot(x,theta))/(1.+e**(np.dot(x,theta)))
    def Train(self,x=None,y=None,iters=500,eps=0.01,l1=False,l2=False,lamda=0.01):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if l1==False and l2==False:
            for iter in xrange(0,iters):
                for i in xrange(0,x.shape[0]):
                    for j in xrange(0,x.shape[1]):
                        delta=x[i,j]*(self.h(x[i,j],self.theta[j])-y[i][0])
                        self.theta[j]=self.theta[j] - (eps* delta)
                    print self.theta
        elif l1==True:
            for iter in xrange(0,iters):
                for i in xrange(0,x.shape[0]):
                    for j in xrange(0,x.shape[1]):
                        pred=x[i,j]*self.theta[j]
                        delta=x[i,j]*(pred-y[i][0])
                        self.theta[j]=self.theta[j] - (eps* delta)
                    print self.theta
        elif l2==True:
            for iter in xrange(0,iters):
                for i in xrange(0,x.shape[0]):
                    for j in xrange(0,x.shape[1]):
                        pred=x[i,j]*self.theta[j]
                        delta=x[i,j]*(pred-y[i][0])
                        self.theta[j]=self.theta[j] - (eps* delta)+lamda*eps*self.theta[j]
                    print self.theta
        return self.theta

    def Predict(self,x=None,y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        theta=self.Train(x,y)
        return self.h(x,self.theta)
        #return [self.h(x,self.theta)>0.5]


datasetx=np.array([[1,0],[1,1],[0,1],[0,0]])
datasety=np.array([[1],[1],[1],[0]])

lg=LogisticRegressionSGD(datasetx,datasety)
lg.Train(l2=True)
print lg.Predict()