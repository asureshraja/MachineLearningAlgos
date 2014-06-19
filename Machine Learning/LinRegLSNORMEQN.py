import numpy as np

#Batch Gradient Descent
class LinearRegressionBGD():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.theta=np.zeros(shape=(x.shape[1],1))

    def Train(self,x=None,y=None,iter=100,eps=0.01,reg=False,lamda=0.001):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if reg==False:
            self.theta=np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))
        else:
            self.theta=np.dot(np.linalg.inv(np.dot(x.T,x)+lamda*np.identity(x.shape[1])),np.dot(x.T,y))

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
print lg.Train(reg=True)
print lg.Predict()