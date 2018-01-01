# -*- coding: utf-8 -*-

"""
This is similar to train.py and train2.py
but now includes covolutional and pooling layers

image size=110x134
"""

import tensorflow as tf
import numpy as np
import sys

#if len(sys.argv) < 2:
#    print("ERROR: You need to enter the file name with the data.")
#    sys.exit(1)
#else:
#    file_name=sys.argv[1]

file_name='clean_training_data.npy'

tdata=np.load(file_name)


#tdata=np.load('new_training_data.npy')

#Flatten input and output data.    

x_data=tdata[:,0]
y_data=tdata[:,1]
new_x_data=[]
new_y_data=[]
for element in x_data:
    new_x_data.append(element.flatten())
    #new_x_data.append(element)
x_data=new_x_data
for element in y_data:
    new_y_data.append(element)
y_data=new_y_data

#Create tensorflow model with placeholders, variables, 
#activation function, error funtion and training function.

x = tf.placeholder(tf.float32, [None, 14740], name='input_tensor')

W_conv1=tf.Variable(tf.random_normal([5,5,1,32]), name='W_conv1')
W_conv2=tf.Variable(tf.random_normal([5,5,32,64]), name='W_conv2')

#W_dense = tf.Variable(tf.random_normal([7*7*64, 10]), name='W_dense')
dimension=(int(28*34*64))
W_dense = tf.Variable(tf.random_normal([dimension, 10]), name='W_dense')
W_out = tf.Variable(tf.random_normal([10, 4]), name='W_out')

b_conv1 = tf.Variable(tf.random_normal([32]), name='b_conv1')
b_conv2 = tf.Variable(tf.random_normal([64]),name='b_conv2')
b_dense = tf.Variable(tf.random_normal([10]), name='b_dense')
b_out = tf.Variable(tf.random_normal([4]), name='b_out')

x_reshaped = tf.reshape(x, shape=[-1, 110, 134, 1], name='x_reshaped')

conv1 = tf.nn.relu(tf.nn.conv2d(x_reshaped, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1, name='conv1')
conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='conv1_1')
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2, name='conv2')
conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='conv2_1')


#need to reshape output of conv2
#conv_out = tf.reshape(conv2, shape=[-1, 7*7*64])
conv_out = tf.reshape(conv2, shape=[-1, dimension])
dense = tf.nn.softmax(tf.matmul(conv_out, W_dense) + b_dense)
y = tf.nn.softmax(tf.matmul(dense, W_out) + b_out)
y_ = tf.placeholder(tf.float32, [None, 4])

#y_before_activation=tf.matmul(y,W_out)+ b_out
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
init = tf.global_variables_initializer()

#These will be used for graphing data 
#(cross entropy/error and weights to see 
#how the training progressed.
#Replace later with Tensorboard.

c_all=[]
w_all=[]


sess = tf.Session()
sess.run(init)

#Some initial data just to make sure the model looks right.

print("A Sample Weight:", sess.run(W_conv1[1,2]))
print("Size of input tensor (x_reshaped):\n", x_reshaped.get_shape())
print("Shape of conv1:\n", conv1.get_shape())
print("Shape of conv2:\n", conv2.get_shape())
print("Size of reshaped tensor:\n", conv_out.get_shape())

print("Size of Weight tensor W_conv1:\n", W_conv1.get_shape())
print("Size of Training Output Tensor:\n", y_.get_shape())
print("Size of Actual Output Tensor:\n", y.get_shape())

EPOCHS=200
batch_size=10

for i in range(EPOCHS):
    
    m=0
    
    print("Epoch:", i)

#Batching is done here.
        
    while m<len(x_data):
      start=m
      end=m+batch_size
      end_of_last_batch=len(x_data)-len(x_data)%batch_size

      batch_x=np.array(x_data[start:end])
      batch_y=np.array(y_data[start:end])
  
      _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y})
      print ("Runing")
      print("Cross Entropy: ", c)
      
      if (m==end_of_last_batch):
          c_all.append(c)
          print("Cross Entropy: ", c)
          #Uncomment to get a weght as it changes.
          #Processor intesive.
          print ("A Sample Weight:", sess.run(W_dense[5000,2]))
          print ("A Sample Weight:", sess.run(W_out[5,2]))
      m+=batch_size

#Graphical output at the end of training
#to provide a visual indication that all is well.
      
import numpy as np
import matplotlib.pyplot as plt

#Since there are four outputs,
#the following divides the input nodes into quarters,
#then reshapes each to show tham as an image,
#each image corresponds to each of the four outputs

for i in range (4):
    a=sess.run(W_dense[0:14740, i])
    image=np.reshape(a, (110,134))
    print(i)
    plt.imshow(image)
    plt.show()

#Calculate the accuracy of training
#Need to segregate training and testing data    
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y}))

#Plot the error fuction over the trianing iterations

plt.plot(c_all, 'x')
plt.show()

#Save the weights manually
weights=sess.run(W_conv1)
file_name = 'weights2.npy'
np.save(file_name,weights)

#Save the weights using TensorFlow trainer
saver = tf.train.Saver()
#Prepend with ./ for Windows directory issue
save_path = saver.save(sess, "./tmp/model2.ckpt")
print("Model saved in file: %s" % save_path)


