# -*- coding: utf-8 -*-

"""


"""

import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) < 2:
    print("ERROR: You need to enter the file name with the data.")
    sys.exit(1)
else:
    file_name=sys.argv[1]

tdata=np.load(file_name)


#tdata=np.load('new_training_data.npy')

#Flatten input and output data.    

x_data=tdata[:,0]
y_data=tdata[:,1]
new_x_data=[]
new_y_data=[]
for element in x_data:
    new_x_data.append(element.flatten())
x_data=new_x_data
for element in y_data:
    new_y_data.append(element)
y_data=new_y_data

#Create tensorflow model with placeholders, variables, 
#activation function, error funtion and training function.

x = tf.placeholder(tf.float32, [None, 14740])
W = tf.Variable(tf.zeros([14740, 4]))
b = tf.Variable(tf.zeros([4]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 4])
y_before_activation=tf.matmul(x,W)+b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
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

print ("A Sample Weight:", sess.run(W[100,2]))
print("Size of input tensor:\n", x.get_shape())
print("Size of Weight tensor:\n", W.get_shape())
print("Size of Training Output Tensor:\n", y_.get_shape())
print("Size of Actual Output Tensor:\n", y.get_shape())

EPOCHS=2000
batch_size=100

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
      
      if (m==end_of_last_batch):
          c_all.append(c)
          print("Cross Entropy: ", c)
          #Uncomment to get a weght as it changes.
          #Processor intesive.
          #print ("A Sample Weight:", sess.run(W[100,2]))
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
    a=sess.run(W[0:14740, i])
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
weights=sess.run(W)
file_name = 'weights.npy'
np.save(file_name,weights)

#Save the weights using TensorFlow trainer
saver = tf.train.Saver()
#Prepend with ./ for Windows directory issue
save_path = saver.save(sess, "./tmp/model.ckpt")
print("Model saved in file: %s" % save_path)


