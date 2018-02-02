''' generative adverserial network serve the need to produce
real type data which can be used for further investigation
'''
''' 
 import all the dependencies ....
 we are going to code the basic model in keras 
 and we are going to visualize it in tensorboard
'''
from data_maker import *

import matplotlib.pyplot as plt
plt.ion()
input_data_size = 3000
data = data(input_data_size)
data = convert_data(data)
hidden_size = data.shape[1]

import tensorflow as tf

# Discriminator Net
initializer = tf.contrib.layers.xavier_initializer()
X = tf.placeholder(tf.float32, shape=[None, hidden_size], name='X')

D_W1 = tf.Variable(initializer([hidden_size, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(initializer([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, hidden_size], name='Z')

G_W1 = tf.Variable(initializer([hidden_size, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(initializer([128, hidden_size]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[hidden_size]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

sess = tf.Session()
m = []
i = []
for it in range(1000):
    sess.run(tf.global_variables_initializer())
    X_mb = data
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                              X: X_mb, Z: sample_Z(data.shape[0], data.shape[1])})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                              Z: sample_Z(data.shape[0], data.shape[1])})
    if it % 10 == 0:
        measure = D_loss_curr + np.abs(0.5 * D_loss_curr - G_loss_curr)
        print('Iter-{}; Convergence measure: {:.4}'.format(it, measure))
        m.append(measure)
        i.append(it)
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(1, data.shape[1])})
        print((samples))
        ''' plt.plot(it)
        plt.pause(0.05) '''
    #print(D_loss_curr,G_loss_curr)

''' #print(data.shape[0],data.shape[1])
while True:
    plt.pause(0.05)
 '''
