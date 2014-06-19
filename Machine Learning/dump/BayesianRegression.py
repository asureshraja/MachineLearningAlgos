import numpy as np

#Batch Gradient Descent
class BayesianLinearRegression():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.theta=np.zeros(shape=(x.shape[1],1))


    def Train(self,x=None,y=None,varOfPrior=5,varOfLikelihoodSigma=0.001):

        if x is None:
            x = self.x
        if y is None:
            y = self.y

        self.theta=np.dot(np.linalg.inv(np.dot(x.T,x)+(varOfLikelihoodSigma * np.linalg.inv(varOfPrior* np.identity(x.shape[1])))),np.dot(x.T,y))

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

lg=BayesianLinearRegression(datasetx,datasety)
print lg.Train()
print lg.Predict()