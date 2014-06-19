import numpy as np

#Batch Gradient Descent
class LinearRegressionBGD():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.theta=np.zeros(shape=(x.shape[1]))

    def Train(self,x=None,y=None,iters=100,eps=0.01,reg=False,lamda=0.01):
        if x is None:
            x = self.x
        if y is None:
            y = self.y


        if reg==False:
            for iter in xrange(0,iters):
                for j in xrange(0,x.shape[1]):
                    delta=0
                    for i in xrange(0,x.shape[0]):
                        pred=x[i,j]*self.theta[j]
                        delta=delta+(x[i,j]*(pred-y[i][0]))
                    self.theta[j]=self.theta[j] - (eps* delta)
                    print self.theta
        else:
            for iter in xrange(0,iters):
                for j in xrange(0,x.shape[1]):
                    delta=0
                    for i in xrange(0,x.shape[0]):
                        pred=x[i,j]*self.theta[j]
                        delta=delta+(x[i,j]*(pred-y[i][0]))
                    self.theta[j]=self.theta[j] - (eps* delta)+lamda*self.theta[j]
                    print self.theta

        return self.theta

    def Predict(self,x=None,y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        theta=self.Train(x,y)
        return np.dot(x,self.theta)


datasetx=np.array([[1,0],[1,1],[0,1],[0,0]])
datasety=np.array([[1],[1],[1],[0]])

lg=LinearRegressionBGD(datasetx,datasety)
lg.Train()
print lg.Predict()