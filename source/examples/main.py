"""
jon dodge
"""
from __future__ import division
from __future__ import print_function

import cPickle
import numpy as np
from MLP import MultiLayerPerceptron

def EvalMLPwithParams(num_epochs, hidden_units, batchSize, learning_rate, momentum, l2_penalty, train_x, train_y, test_x, test_y):
    print("\n\n********* Run Parameters ******************")
    print("numEpochs\t", num_epochs)
    print("hidden_units\t", hidden_units)
    print("batchSize\t", batchSize)
    print("learning_rate\t", learning_rate)
    print("momentum\t", momentum)
    print("l2_penalty\t", l2_penalty)

    num_examples, input_dims = train_x.shape
    mlp = MultiLayerPerceptron(input_dims, hidden_units)

    print("\n\t\tEpoch\tTrain Loss\tTest Loss\tTrain Acc\tTest Acc")
    for epoch in xrange(num_epochs):
        train_loss = test_loss = train_accuracy = test_accuracy = 0.0

        # shuffle the training data, ensuring X's stay paired with appropriate Y
        trainWhole = zip(train_x, train_y)
        np.random.shuffle(trainWhole)
        train_x, train_y = zip(*trainWhole)
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        # train on each minibatch
        oldGradients = None
        for i in range(0, num_examples, batchSize):
            minibatchX = train_x[i:i + batchSize]
            minibatchY = train_y[i:i + batchSize]
            oldGradients = mlp.train(minibatchX, minibatchY, learning_rate, momentum, l2_penalty, oldGradients)

        # training on mini batches complete, lets test the model for this epoch (first on the TEST data)
        for (x, y) in zip(test_x, test_y):
            (correct, lossHere) = mlp.evaluate(x, y)
            test_accuracy += correct
            test_loss += lossHere
        test_accuracy /= len(test_y)
        test_loss /= len(test_y)

        # now test on the TRAIN data
        for (x, y) in zip(train_x, train_y):
            (correct, lossHere) = mlp.evaluate(x, y)
            train_accuracy += correct
            train_loss += lossHere
        train_accuracy /= len(train_y)
        train_loss /= len(train_y)

        # And finally, some output
        print('\t\t', epoch, '\t', '{0:.3f}'.format(train_loss), '\t', '{0:.3f}'.format(test_loss), '\t', train_accuracy, '\t',
              test_accuracy)

if __name__ == '__main__':
    print('Loading Data...', ),
    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
    # normalize the image data
    train_x = train_x / 255.0
    print('Data loaded!')

    # specify parameters (as a list).  FIRST will be the "base" number, then variations
    num_epochs = 50
    hiddenUnitsList = [54, 4, 14, 104, 154, 254, 554]
    batchSizeList = [53, 3, 13, 103, 153, 253, 553, 1503, 3003, 4003]
    learningRateList = [.0001, .001, .01, .05, .1, .5, 1, 5, 10, 50]
    momentumList = [.3, 0, .5, .8]
    l2PenaltyList = [.01, 0, .05, .1, .5, 1, 5, 10, 50]


    # this call is a pretty well performing example
    EvalMLPwithParams(num_epochs, hiddenUnitsList[0], batchSizeList[0],
                      learningRateList[1], momentumList[2], l2PenaltyList[2],
                      train_x, train_y, test_x, test_y)

    # perform grid search on parameters, keeping others fixed
    '''
    for hu in hiddenUnitsList:
        EvalMLPwithParams(num_epochs, hu, batchSizeList[0],
                          learningRateList[0], momentumList[0], l2PenaltyList[0],
                          train_x, train_y, test_x, test_y)
    '''
    '''
    for bs in batchSizeList:
        EvalMLPwithParams(num_epochs, hiddenUnitsList[0], bs,
                          learningRateList[0], momentumList[0], l2PenaltyList[0],
                          train_x, train_y, test_x, test_y)
    '''
    '''
    for lr in learningRateList:
        EvalMLPwithParams(num_epochs, hiddenUnitsList[0], batchSizeList[0],
                          lr, momentumList[0], l2PenaltyList[0],
                          train_x, train_y, test_x, test_y)
    '''
    '''
    for mom in momentumList:
        EvalMLPwithParams(num_epochs, hiddenUnitsList[0], batchSizeList[0],
                          learningRateList[0], mom, l2PenaltyList[0],
                          train_x, train_y, test_x, test_y)
    '''
    '''
    for lp in l2PenaltyList:
        EvalMLPwithParams(num_epochs, hiddenUnitsList[0], batchSizeList[0],
                          learningRateList[0], momentumList[0], lp,
                          train_x, train_y, test_x, test_y)
    '''