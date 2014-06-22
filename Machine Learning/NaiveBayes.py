'''
Naive Bayes

@author: suresh
'''
import numpy as np
import math

def normpdf(x, mean, var):
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

class NaiveBayes():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.means=[]
        self.meanforfeatures()
        self.vars=[]
        self.varforfeatures()
        self.priors=[]
        self.priorsofclasses()

    def meanforfeatures(self):
        for i in np.unique(self.y):
            self.means.append(np.mean(self.x[self.y[:,0]==i],0))
        print self.means
        
    def varforfeatures(self):
        for i in np.unique(self.y):
            self.vars.append(np.var(self.x[self.y[:,0]==i],0))
        print self.vars
        
    def priorsofclasses(self):
        for i in np.unique(self.y):
            self.priors.append(float(len(self.x[self.y[:,0]==i]))/len(self.x))

        
    def likelihoodswithpriors(self,testx):
        for t in testx:
            templiks=[]
            for i in xrange(0,len(np.unique(self.y))):
                temp=1;
                for j in xrange(0,self.x.shape[1]):
                    print normpdf(t[j], self.means[i][j], self.vars[i][j]),self.means[i][j],self.vars[i][j]
                    temp=temp*normpdf(t[j], self.means[i][j], self.vars[i][j])
                temp=temp*self.priors[i]
                templiks.append(temp)
            self.likelihoods.append(templiks)
        print self.likelihoods
        
    def Predict(self,testx):
        self.likelihoods=[]
        self.likelihoodswithpriors(testx)
        for val in self.likelihoods:
            print np.unique(self.y)[val.index(max(val))]
        
datasetx=np.array([[6,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])
datasety=np.array([[0],[0],[0],[0],[2],[2],[2],[2]])
nb=NaiveBayes(datasetx,datasety)
testx=np.array([[6,130,8]])
nb.Predict(testx)
