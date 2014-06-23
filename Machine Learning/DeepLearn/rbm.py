import numpy as np
float_formatter = lambda x: "%.2f" %x
np.set_printoptions(formatter={'float_kind':float_formatter})

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

class rbm:
    def __init__(self,input,noofvisible,noofhidden,rnseed=123):
        self.input=input
        self.noofvisible=noofvisible
        self.noofhidden=noofhidden
        self.rng=np.random.RandomState(rnseed)
        temp=1/noofvisible
        self.weights=self.rng.uniform(low=-temp,high=temp,size=(noofvisible,noofhidden))
        self.hbias=np.zeros(noofhidden)
        self.vbias=np.zeros(noofvisible)

    def hiddenprobs(self,visiblevalues):
        return np.dot(visiblevalues,self.weights)

    def visibleprobs(self,hiddenvalues):
        return np.dot(hiddenvalues,self.weights.T)

    def hiddenactivations(self,visiblevalues):
        return sigmoid(self.hiddenprobs(visiblevalues)+self.hbias)

    def visibleactivations(self,hiddenvalues):
        return sigmoid(self.visibleprobs(hiddenvalues)+self.vbias)

    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        pos_hidden_activations=self.hiddenactivations(self.input)
        pos_hidden_samples=self.rng.binomial(size=pos_hidden_activations.shape,n=1,p=pos_hidden_activations)


        chain_start=pos_hidden_samples
        for step in xrange(k):
            if step == 0:
                neg_visible_activations=self.visibleactivations(chain_start)
                neg_visible_samples=self.rng.binomial(size=neg_visible_activations.shape,n=1,p=neg_visible_activations)
                neg_hidden_activations=self.hiddenactivations(neg_visible_samples)
                neg_hidden_samples=self.rng.binomial(size=neg_hidden_activations.shape,n=1,p=neg_hidden_activations)
            else:
                neg_visible_activations=self.visibleactivations(neg_hidden_samples)
                neg_visible_samples=self.rng.binomial(size=neg_visible_activations.shape,n=1,p=neg_visible_activations)
                neg_hidden_activations=self.hiddenactivations(neg_visible_samples)
                neg_hidden_samples=self.rng.binomial(size=neg_hidden_activations.shape,n=1,p=neg_hidden_activations)



        self.weights+=lr*(np.dot(self.input.T,pos_hidden_samples)-np.dot(neg_visible_samples.T,neg_hidden_activations))
        self.vbias += lr * np.mean(self.input - neg_visible_samples, axis=0)
        self.hbias += lr * np.mean(pos_hidden_samples - neg_hidden_activations, axis=0)
        print self.weights

    def reconstruct(self, visiblevalues):
        return self.visibleactivations(self.hiddenactivations(visiblevalues))

def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = np.array([[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]])


    rng = np.random.RandomState(123)

    # construct RBM
    r= rbm(data, 6, 2,123)

    # train
    for epoch in xrange(training_epochs):
        r.contrastive_divergence(lr=learning_rate, k=k)
        # cost = rbm.get_reconstruction_cross_entropy()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    # test
    v = np.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])

    print r.reconstruct(v)



if __name__ == "__main__":
    test_rbm()