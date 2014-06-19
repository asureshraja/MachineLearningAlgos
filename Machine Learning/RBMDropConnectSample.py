#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Suresh
#
# Created:     16/01/2014
# Copyright:   (c) Suresh 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import csv
import pickle
float_formatter = lambda x: "%.2f" %x


def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def linear(x):
    return x



def main():
    np.set_printoptions(formatter={'float_kind':float_formatter})
    num_visible=3952
    num_hidden=12
    learning_rate=0.001
    max_epochs=20
    weights = 0.1 * np.random.randn(num_visible, num_hidden)
    weights = np.insert(weights, 0, 0, axis = 0)
    weights = np.insert(weights,0,0,axis=1)

    reader=csv.reader(open("C:\\Users\\Suresh\\Desktop\\MTECHPROJECT\\phase2\\newmodel2\\datasets\\train.csv","rb"),delimiter=',')
    x=list(reader)
    dataset=np.array(x).astype('float')

    count=0



#training procedures
    for epoch in range(max_epochs):
        for data in dataset:
            count=0
            data=np.array([data])
            num_examples = data.shape[0]
            data = np.insert(data, 0, 1, axis = 1)
            pos_hidden_activations=np.dot(data, weights)
            pos_hidden_probs = logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > ((np.random.rand(num_examples, num_hidden + 1)*0)+0.5)
            pos_associations = np.dot(data.T, pos_hidden_probs)

            neg_visible_activations = np.dot(pos_hidden_states, weights.T)
            neg_visible_probs = linear(neg_visible_activations)
            neg_visible_probs[:,0] = 1
            neg_hidden_activations = np.dot(neg_visible_probs, weights)
            neg_hidden_probs = logistic(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            delta = learning_rate * ((pos_associations - neg_associations) / num_examples)

            for d in data[0]:

                if int(d)==0:
                    #print("%.2f" % delta[count][1])
                    for h in range(num_hidden+1):
                        delta[count][h]=0
                count=count+1
            weights += delta

            error = np.mean(np.abs(data - neg_visible_probs))
            #print "Epoch %s: error is %s" % (epoch, error)
            print epoch ,"iteration"
            #print neg_visible_prob
            bigsum=0
            for a in range(data.shape[0]):
                sum=0
                c=0
                for i,j in zip(data[a],neg_visible_probs[a]):
                    if(i!=0):
                        #print i,j,i-round(j)
                        c=c+1
                        sum=sum+abs(i-round(j))
                bigsum=bigsum+sum/c
            print bigsum/data.shape[0]


    pickleFile = open("C:\\Users\\Suresh\\Desktop\\MTECHPROJECT\\phase2\\newmodel2\\binaries\\fullweights12-20.bin", 'wb')
    pickle.dump(weights, pickleFile, pickle.HIGHEST_PROTOCOL)
    pickleFile.close()
if __name__ == '__main__':
    main()
