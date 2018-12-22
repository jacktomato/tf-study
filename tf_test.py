import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()  #deprecated
#create tensorflow structure end

sess = tf.Session()
sess.run(init) #very important

for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step,sess.run(Weights),sess.run(biases))
'''
def add_layer(inputs,in_size,out_size,activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size])+0.1)
	Wx_plus_b = tf.matmul(tf.cast(inputs,tf.float32),Weights)+biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(x_data,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
	reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(10000):
	sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
	if i % 50 == 0:
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		#print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
		prediction_value = sess.run(prediction,feed_dict={xs:x_data})
		lines = ax.plot(x_data,prediction_value,'r-',lw=5)
		plt.pause(0.1 )