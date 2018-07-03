import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import gym
import math


#num_inp = 784
num_inp = 9
num_out = 10
batch_size = 1
num_nodes = 5
epochs = 5

def weight_list(num_inp,num_nodes,num_time_steps):
	weights = []
	for i in range(num_time_steps):
		if(i==0):
			weights.append(tf.Variable(tf.truncated_normal([num_inp,num_nodes],stddev=.1)))
		else:
			weights.append(tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1)))
	return weights

def bias_list(num_inp,num_nodes,num_time_steps):
	bias = []
	for i in range(num_time_steps):
		bias.append(tf.Variable(tf.truncated_normal([1,num_nodes],stddev=.1)))
	return bias

def rnn_graph(w,b,x):
	y_comp = 0
	y_layer = []
	for i in range(len(w)):
		if(i==0):
			y_comp = tf.tanh(tf.matmul(x,w[i]) + b[i])
		else:
			y_comp = tf.tanh(tf.matmul(y_layer[i-1],w[i]) + b[i] + y_comp)
		y_layer.append(y_comp)		
	return y_comp

def add_layers(nodes_per_lay,num_lay,lay_1):
	w = tf.Variable(tf.random_uniform([nodes_per_lay,nodes_per_lay]))
	b = tf.Variable(tf.random_uniform([nodes_per_lay]))
	y = tf.nn.relu(tf.matmul(lay_1,w)+b)
	if num_lay == 0:
		return y
	else:
		return add_layers(nodes_per_lay,num_lay-1,y)

def convolutional_layer(x,shape):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	b = tf.Variable(tf.truncated_normal([shape[3]],stddev=0.1))
	conv = tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')+b
	act_fun = tf.nn.relu(conv)
	drop = tf.nn.dropout(act_fun,keep_prob=0.5)
	return drop
	
def max_pool(x,pool_shape,stride_shape):
	return tf.nn.max_pool(x,ksize=pool_shape,strides=stride_shape,padding='SAME')


layer = 0
height = 3
width = 3
for i in range():
	for j in range()
		ap = multiply(big[i+h][j],W) + b
		layer_t.append()
		

def big_lstm(batch_size,num_nodes,x_old,h_old,c_old):
	W_f = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	W_i = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	W_o = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	W_c = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	W_a = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))

	U_f = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	U_i = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	U_o = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	U_c = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))
	U_a = tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1))

	B_f = tf.Variable(tf.truncated_normal([num_nodes],stddev=.1))
	B_i = tf.Variable(tf.truncated_normal([num_nodes],stddev=.1))
	B_o = tf.Variable(tf.truncated_normal([num_nodes],stddev=.1))
	B_a = tf.Variable(tf.truncated_normal([num_nodes],stddev=.1))

	F_t = tf.sigmoid( tf.matmul(x_old,W_f) + tf.matmul(h_old,U_f) + B_f)
	I_t = tf.sigmoid( tf.matmul(x_old,W_i) + tf.matmul(h_old,U_i) + B_i)
	O_t = tf.sigmoid( tf.matmul(x_old,W_o) + tf.matmul(h_old,U_o) + B_o)
	A_t = tf.tanh(    tf.matmul(x_old,W_a) + tf.matmul(h_old,U_a) + B_a)

	c_t = tf.multiply(I_t,c_old) + tf.multiply(F_t,A_t)
	h_t = tf.multiply(O_t,tf.tanh(c_t))

	return c_t,h_t

def small_lstm(num_inp,num_nodes,x_old,h_old,c_old):
	W_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	W_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	W_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	W_c = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	W_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

	U_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	U_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	U_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	U_c = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	U_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

	B_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	B_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	B_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
	B_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

	F_t = tf.sigmoid( tf.multiply(x_old,W_f) + tf.multiply(h_old,U_f) + B_f)
	I_t = tf.sigmoid( tf.multiply(x_old,W_i) + tf.multiply(h_old,U_i) + B_i)
	O_t = tf.sigmoid( tf.multiply(x_old,W_o) + tf.multiply(h_old,U_o) + B_o)
	A_t = tf.tanh(    tf.multiply(x_old,W_a) + tf.multiply(h_old,U_a) + B_a)

	c_t = tf.multiply(I_t,c_old) + tf.multiply(F_t,A_t)
	h_t = tf.multiply(O_t,tf.tanh(c_t))

	return c_t,h_t

def get_data(num_inp):
	#get random number
	offset = np.random.uniform(low=-1,high=1,size=1)

	#Define data
	x = []
	y = []
	for i in range(num_inp):
		x.append(.1*i) #+offset[0])
		y.append(math.sin(.1*i))#+offset[0]))
	#x = np.array(x).reshape(1,num_inp)
	#y = np.array(y).reshape(1,num_inp)
	return x,y

#Define placeholders
x_old = tf.placeholder(tf.float32,shape=[num_inp])
y_true = tf.placeholder(tf.float32,shape=[num_inp])

#Create graph
h0 = 0.0
c0 = 0.0

c1,h1 = small_lstm(batch_size,num_nodes,x_old[0],h0,c0)
c2,h2 = small_lstm(batch_size,num_nodes,x_old[0],h1,c1)
c3,h3 = small_lstm(batch_size,num_nodes,x_old[0],h2,c2)
c4,h4 = small_lstm(batch_size,num_nodes,x_old[0],h0,c0)
c5,h5 = small_lstm(batch_size,num_nodes,x_old[0],h1,c1)
c6,h6 = small_lstm(batch_size,num_nodes,x_old[0],h2,c2)
c7,h7 = small_lstm(batch_size,num_nodes,x_old[0],h0,c0)
c8,h8 = small_lstm(batch_size,num_nodes,x_old[0],h1,c1)
c9,h9 = small_lstm(batch_size,num_nodes,x_old[0],h2,c2)

h_list = [h1,h2,h3,h4,h5,h6,h7,h8,h9]
c_list = [c1,c2,c3,c4,c5,c6,c7,c8,c9]

#Step 4) Loss Function
num = 0
for i in range(len(h_list)):
	var1 = tf.subtract(y_true[i],h_list[i])
	var2 = var1*var1
	num+=var2
cost1 = (1/num_inp)*(num)

#Step 5) Create optimizer
optimizer = tf.train.GradientDescentOptimizer(.01).minimize(cost1)

#Step 6) Create Session
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for ep in range(epochs):
		for steps in range(30000):
			batch_x,batch_y = get_data(num_inp)

			sess.run(optimizer,feed_dict={x_old: batch_x , y_true: batch_y})
	
		print(sess.run(cost1,feed_dict={x_old: batch_x , y_true: batch_y}))

		batch_x, batch_y = get_data(num_inp)

		plt.scatter(batch_x,batch_y)
		plt.scatter(batch_x,sess.run(h_list,feed_dict={x_old: batch_x , y_true: batch_y}))
		plt.show()


