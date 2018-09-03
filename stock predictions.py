# Import
import tensorflow as tensorflow
import numpy as numpy
import pandas as pandas
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plot

# Import data
input = pandas.read_csv('01_input/input_stocks.csv')

# Drop date variable
input = input.drop(['DATE'], 1)

# Dimensions of inputset
n = input.shape[0]
p = input.shape[1]

# Make input a numpy.array
input = input.values

# Training and test data
train_start = 0
train_end = int(numpy.floor(0.8*n))
test_start = train_end + 1
test_end = n
input_train = input[numpy.arange(train_start, train_end), :]
input_test = input[numpy.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(input_train)
input_train = scaler.transform(input_train)
input_test = scaler.transform(input_test)

# Build X and y
X_train = input_train[:, 1:]
y_train = input_train[:, 0]
X_test = input_test[:, 1:]
y_test = input_test[:, 0]

# Number of stocks in training input
n_stocks = X_train.shape[1]

# Neurons
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Session
net = tensorflow.InteractiveSession()

# Placeholder
X = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, n_stocks])
Y = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tensorflow.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tensorflow.zeros_initializer()

# Hidden weights
wh_1 = tensorflow.Variable(weight_initializer([n_stocks, n_neurons_1]))
bh_1 = tensorflow.Variable(bias_initializer([n_neurons_1]))
wh_2= tensorflow.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bh_2 = tensorflow.Variable(bias_initializer([n_neurons_2]))
wh_3= tensorflow.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bh_3 = tensorflow.Variable(bias_initializer([n_neurons_3]))
wh_4 = tensorflow.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bh_4 = tensorflow.Variable(bias_initializer([n_neurons_4]))

# outputput weights
W_output = tensorflow.Variable(weight_initializer([n_neurons_4, 1]))
bias_output = tensorflow.Variable(bias_initializer([1]))

# Hidden layer
h_1 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(X, wh_1), bh_1))
h_2 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(h_1, W_h_2), bh_2))
h_3 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(h_2, W_h_3), bh_3))
h_4 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(h_3, wh_4), bh_4))

# outputput layer (transpose!)
output = tensorflow.transpose(tensorflow.add(tensorflow.matmul(h_4, W_output), bias_output))

# Cost function
co = tensorflow.reduce_mean(tensorflow.squared_difference(output, Y))

# Optimizer
opt = tensorflow.train.AdamOptimizer().minimize(co)

# Init
net.run(tensorflow.global_variables_initializer())

# Setup plot
plot.ion()
fig = plot.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plot.show()

# Fit neural net
s = 256
co_t = []
co_test = []

# Run
epochs = 10
for e in range(epochs):

    # Shuffle training input

    shuffle_indices = numpy.random.permutation(numpy.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // s
):
        start = i * s
    
        b_x = X_train[start:start + s
    ]
        b_y = y_train[start:start + s
    ]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: b_x, Y: b_y})

        # Show progress
        if numpy.mod(i, 50) == 0:
            # co train and test
            co_t.append(net.run(co, feed_dict={X: X_train, Y: y_train}))
            co_test.append(net.run(co, feed_dict={X: X_test, Y: y_test}))
            print('co Train: ', co_t[-1])
            print('co Test: ', co_test[-1])
            # pict
            p = net.run(output, feed_dict={X: X_test})
            line2.set_yinput
        (p)
            plot.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plot.pause(0.01)
