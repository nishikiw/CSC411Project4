import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import sys

env = gym.make('CartPole-v0')

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

alpha = 1e-6
gamma = 0.99

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
w = tf.get_variable("w", shape=[input_shape, output_units])
b = tf.get_variable("b", shape=[output_units])
x = tf.placeholder(tf.float32, shape=(None, input_shape), name='x')
y = tf.placeholder(tf.int32, shape=(None, 1), name='y')

layer1 = tf.sigmoid(tf.matmul(x, w)+b)
soft_max = tf.nn.softmax(layer1)

pi_sample = tf.argmax(soft_max, axis=1)
log_pi = tf.log(soft_max)
act_pi = tf.matmul(tf.expand_dims(log_pi, 1), tf.one_hot(y, 2, axis=1))

Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * act_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
for ep in range(5000):
    obs = env.reset()

    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        #env.render()
        action = sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break

    returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
    index = ep % MEMORY
    
    
    _ = sess.run([train_op],
                feed_dict={x:np.array(ep_states),
                            y:np.reshape(np.array(ep_actions), (len(ep_actions), 1)),
                            Returns:returns })
    
    if ep % 100 == 0:
        track_returns.append(G)
        track_returns = track_returns[-MEMORY:]
        mean_return = np.mean(track_returns)
        print("Episode {} finished after {} steps with return {}".format(ep, t, G))
        print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                                mean_return))