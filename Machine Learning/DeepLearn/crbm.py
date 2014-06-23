import sys
import numpy as np
import rbm


class CRBM(rbm.rbm):
    def propdown(self, h):
        pre_activation = np.dot(h, self.W.T) + self.vbias
        return pre_activation



    def sample_v_given_h(self, h0_sample):
        a_h = self.visibleprobs(h0_sample)+self.vbias
        print a_h
        en = np.exp(-a_h)
        ep = np.exp(a_h)

        v1_mean = 1 / (1 - en) - 1 / a_h
        U = np.array(self.rng.uniform(
            low=0,
            high=1,
            size=v1_mean.shape))

        v1_sample = np.log((1 - U * (1 - ep))) / a_h

        return [v1_mean, v1_sample]

    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        pos_hidden_activations=self.hiddenactivations(self.input)
        pos_hidden_samples=self.rng.binomial(size=pos_hidden_activations.shape,n=1,p=pos_hidden_activations)


        chain_start=pos_hidden_samples
        for step in xrange(k):
            if step == 0:
                neg_visible_activations=self.sample_v_given_h(chain_start)[0]
                neg_visible_samples=self.sample_v_given_h(chain_start)[1]
                neg_hidden_activations=self.hiddenactivations(neg_visible_samples)
                neg_hidden_samples=self.rng.binomial(size=neg_hidden_activations.shape,n=1,p=neg_hidden_activations)
            else:
                neg_visible_activations=self.sample_v_given_h(chain_start)[0]
                neg_visible_samples=self.sample_v_given_h(chain_start)[1]
                neg_hidden_activations=self.hiddenactivations(neg_visible_samples)
                neg_hidden_samples=self.rng.binomial(size=neg_hidden_activations.shape,n=1,p=neg_hidden_activations)



        self.weights+=lr*(np.dot(self.input.T,pos_hidden_samples)-np.dot(neg_visible_samples.T,neg_hidden_activations))
        self.vbias += lr * np.mean(self.input - neg_visible_samples, axis=0)
        self.hbias += lr * np.mean(pos_hidden_samples - neg_hidden_activations, axis=0)
        print self.weights


def test_crbm(learning_rate=0.1, k=1, training_epochs=1):
    data = np.array([[0.4, 0.5, 0.5, 0.,  0.,  0.],
                        [0.5, 0.3,  0.5, 0.,  0.,  0.],
                        [0.4, 0.5, 0.5, 0.,  0.,  0.],
                        [0.,  0.,  0.5, 0.3, 0.5, 0.],
                        [0.,  0.,  0.5, 0.4, 0.5, 0.],
                        [0.,  0.,  0.5, 0.5, 0.5, 0.]])


    rng = np.random.RandomState(123)

    # construct CRBM
    crbm = CRBM(data, 6, 2, 123)

    # train
    for epoch in xrange(training_epochs):
        crbm.contrastive_divergence(lr=learning_rate, k=k)
        # cost = rbm.get_reconstruction_cross_entropy()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    # test
    v = np.array([[0.5, 0.5, 0., 0., 0., 0.],
                     [0., 0., 0., 0.5, 0.5, 0.]])

    print crbm.reconstruct(v)


if __name__ == "__main__":
    test_crbm()