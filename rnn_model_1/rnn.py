import os
import sys
import platform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from textwrap import wrap
import re
import itertools
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt

num_input = 28          # MNIST data input (image shape: 28x28)
timesteps = 28          # Timesteps
n_classes = 10          # Number of classes, one class per digit
learning_rate = 0.001   # The optimization initial learning rate
epochs = 10             # Total number of training epochs
batch_size = 100        # Training batch size
display_freq = 100      # Frequency of displaying the training results
num_hidden_units = 128  # Number of hidden units of the RNN
log_dir = os.getcwd()
generic_slash = None
if platform.system() == 'Windows':
  generic_slash = '\\'
else:
  generic_slash = '/'

def encodeLabels(labels_decoded):
    encoded_labels = np.zeros(shape=(len(labels_decoded), n_classes), dtype=np.int8)
    for x in range(0, len(labels_decoded)):
        some_label = labels_decoded[x]

        if 0 == some_label:
            encoded_labels[x][0] = 1
        elif 1 == some_label:
            encoded_labels[x][1] = 1
        elif 2 == some_label:
            encoded_labels[x][2] = 1
        elif 3 == some_label:
            encoded_labels[x][3] = 1
        elif 4 == some_label:
            encoded_labels[x][4] = 1
        elif 5 == some_label:
            encoded_labels[x][5] = 1
        elif 6 == some_label:
            encoded_labels[x][6] = 1
        elif 7 == some_label:
            encoded_labels[x][7] = 1
        elif 8 == some_label:
            encoded_labels[x][8] = 1
        elif 9 == some_label:
            encoded_labels[x][9] = 1
    return encoded_labels

def weight_variable(shape):
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)

def RNN(x, weights, biases, timesteps, num_hidden):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a rnn cell with tensorflow
    rnn_cell = rnn.BasicRNNCell(num_hidden)

    # Get lstm cell output
    # If no initial_state is provided, dtype must be specified
    # If no initial cell state is provided, they will be initialized to zero
    states_series, current_state = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(current_state, weights) + biases


mnist = tf.keras.datasets.mnist
(train_x, train_y),(test_x, test_y) = mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0
total_train_data = len(train_y)
total_test_data = len(test_y)
print("Size of:")
print("- Training-set:\t\t{}".format(total_train_data))
print("- Validation-set:\t{}".format(total_test_data))

print("Encoding Labels")
# One-Hot encode the labels
train_y = encodeLabels(train_y)
test_y = encodeLabels(test_y)

print("Creating Datasets")
# Create the DATASETs
train_x_dataset = tf.data.Dataset.from_tensor_slices(train_x)
train_y_dataset = tf.data.Dataset.from_tensor_slices(train_y)
test_x_dataset = tf.data.Dataset.from_tensor_slices(test_x)
test_y_dataset = tf.data.Dataset.from_tensor_slices(test_y)

print("Zipping The Data Together")
# Zip the data and batch it and (shuffle)
train_data = tf.data.Dataset.zip((train_x_dataset, train_y_dataset)).shuffle(buffer_size=total_train_data).repeat().batch(batch_size).prefetch(buffer_size=5)
test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(batch_size).prefetch(buffer_size=1)

print("Creating Iterators")
# Create Iterators
train_iterator = train_data.make_initializable_iterator()
test_iterator = test_data.make_initializable_iterator()

# Create iterator operation
train_next_element = train_iterator.get_next()
test_next_element = test_iterator.get_next()

# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, timesteps, num_input], name='x')
y_ = tf.placeholder(tf.int8, shape=[None, n_classes], name='y_')

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, n_classes])
# create bias vector initialized as zero
b = bias_variable(shape=[n_classes])

output_logits = RNN(x, W, b, timesteps, num_hidden_units)
y_rnn = tf.nn.softmax(output_logits)

# Define the loss function, optimizer, and accuracy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_logits), name='cross_entropy')
tf.summary.scalar('cross_entropy', cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(cross_entropy)

# Model predictions
prediction_label = tf.argmax(output_logits, axis=1, name='predictions')
actual_label = tf.argmax(y_, 1)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), actual_label, name='correct_pred')

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
tf.summary.scalar('accuracy', accuracy)

# Initialize and Run
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + generic_slash + 'tensorflow' + generic_slash + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + generic_slash + 'tensorflow' + generic_slash + 'test')
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    saver = tf.train.Saver()
    print("----------------------|----\---|-----|----/----\---|-----|---|\----|---------------------")
    print("----------------------|    |---|     ----|     -------|------|-\---|---------------------")
    print("----------------------|   |----|-----|---|   ---------|------|--\--|---------------------")
    print("----------------------|    |---|     ----|     |------|------|---\-|---------------------")
    print("----------------------|----/---|-----|----\----/---|-----|---|----\|---------------------")
    global_counter = 0
    # Number of training iterations in each epoch
    num_tr_iter = int(total_train_data / batch_size)
    for epoch in range(epochs):
        for iteration in range(num_tr_iter):
            print("step " + str(global_counter))
            batch = sess.run(train_next_element)
            x_batch_train = batch[0].reshape((batch_size, timesteps, num_input))
            summary, _ = sess.run([merged, optimizer], feed_dict={x: x_batch_train, y_: batch[1]})
            train_writer.add_summary(summary, global_counter)
            train_writer.flush()
            global_counter += 1
    
        # Run validation after every epoch
        validation_batch = sess.run(test_next_element)
        x_batch_test = validation_batch[0].reshape((batch_size, timesteps, num_input))
        summary, acc = sess.run([merged, accuracy], feed_dict={x: x_batch_test, y_: validation_batch[1]})
        print('epoch ' + str(epoch+1) + ', test accuracy ' + str(acc))
        # Save the model
        saver.save(sess, log_dir + generic_slash + "tensorflow" + generic_slash + "mnist_model.ckpt")
        # Save the summaries
        test_writer.add_summary(summary, global_counter)
        test_writer.flush()

    # Evaluate over the entire test dataset
    # Re-initialize
    test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(total_test_data).prefetch(buffer_size=1)
    test_iterator = test_data.make_initializable_iterator()
    test_next_element = test_iterator.get_next()
    sess.run(test_iterator.initializer)
    
    # Run for final accuracy
    validation_batch = sess.run(test_next_element)
    x_batch_test = validation_batch[0].reshape((total_test_data, timesteps, num_input))

    print('Final Accuracy ' + str(accuracy.eval(feed_dict={
        x: x_batch_test, y_: validation_batch[1]})))
    print("FINISHED")
    
    # Re-initialize
    test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(total_test_data).prefetch(buffer_size=1)
    test_iterator = test_data.make_initializable_iterator()
    test_next_element = test_iterator.get_next()
    sess.run(test_iterator.initializer)

    print("Creating Confusion Matrix")
    validation_batch = sess.run(test_next_element)
    x_batch_test = validation_batch[0].reshape((total_test_data, timesteps, num_input))

    predict, correct = sess.run([prediction_label, actual_label], feed_dict={
        x: x_batch_test, y_: validation_batch[1]})
    skplt.metrics.plot_confusion_matrix(correct, predict, normalize=True)
    plt.savefig(log_dir + generic_slash + "tensorflow" + generic_slash + "plot.png")

