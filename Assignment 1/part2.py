import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from part1 import dist

def loadData():
    np.random.seed(521)
    Data = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
    Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    return trainData, trainTarget, validData, validTarget, testData, testTarget

'''
@parameters: distVector(N1 x N2), K(int32)
@return: (N1 x N2)
'''
def knn(distVector, K=1):
    knnDist, knnIndex = tf.nn.top_k(-distVector, k=K)
    numTrainData = tf.shape(distVector)[1]
    r = tf.reduce_sum(tf.to_float(tf.equal(tf.expand_dims(knnIndex, 2), tf.reshape(tf.range(numTrainData), [1,1,-1]))), 1)
    return r/tf.to_float(K)

'''
@parameters: r(N1 x N2), trainTar(N2 x 1)
@return: (N1 x 1)
'''
def predKnn(r, trainTar):
    return tf.matmul(r, trainTar)

'''
@parameters: y(N1 x 1), y_(N1 x 1)
@return: (float64)
'''
def mse(y, y_):
    return tf.reduce_mean(tf.reduce_sum((y - y_)**2, 1)) / 2

def buildGraph():
    # Variabe Creation
    trainX = tf.placeholder(tf.float32, [None, 1], name="train_x")
    trainY = tf.placeholder(tf.float32, [None, 1], name="train_y")
    newX = tf.placeholder(tf.float32, [None, 1], name="new_x")
    newY = tf.placeholder(tf.float32, [None, 1], name="new_y")
    k = tf.placeholder("int32", name="k")

    # KNN 
    predY = predKnn(knn(dist(newX, trainX), K = k), trainY)
    lossMSE = mse(predY, newY)

    return trainX, trainY, newX, newY, k, predY, lossMSE


if __name__ == '__main__':
    # load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()
    k_choice = [1, 3, 5, 50]

    # load computation_Graph
    trainX, trainY, newX, newY, k, predY, lossMSE = buildGraph()

    # Initialization
    sess = tf.InteractiveSession()
    MSE_result = defaultdict(list)

    # Choose Best k with MSE
    for kc in k_choice:
        mse_valid = sess.run(lossMSE,
                             feed_dict={trainX: trainData, trainY: trainTarget, newX: validData, newY: validTarget,
                                             k: kc})
        MSE_result['validation'].append(mse_valid)

        mse_test = sess.run(lossMSE,
                            feed_dict={trainX: trainData, trainY: trainTarget, newX: testData, newY: testTarget, k: kc})
        MSE_result['test'].append(mse_test)

        print("K=%d\t validation MSE: %f, test MSE: %f" % (kc, mse_valid, mse_test))

    k_best = k_choice[np.argmin(MSE_result['validation'])]
    print("Best k using validation set is k=%d" % k_best)

    plt.clf()
    plt.plot(k_choice, MSE_result['validation'],label='validation MSE')
    plt.plot(k_choice, MSE_result['test'], label='test MSE')
    plt.legend()
    plt.title("validation MSE and test MSE against varying k")
    plt.savefig('part2_MSE.png')

    # Prediction and Regression
    X = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]

    for kc in k_choice:
        prediction = sess.run(predY, feed_dict={trainX:trainData, trainY:trainTarget, newX:X, k:kc})

        plt.clf()
        plt.plot(trainData, trainTarget, '.')
        plt.plot(X, prediction, '-')
        plt.title("k-NN regression on data1D, k=%d" % kc)
        plt.savefig('part2_prediction_k=' + '%g.png' % kc)