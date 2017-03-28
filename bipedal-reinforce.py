#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import sys



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--load-model', metavar='NPZ',
                    help='NPZ file containing model weights/biases')
args = parser.parse_args()



env = gym.make('BipedalWalker-v2')

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

hidden_size = 64
alpha = 0.01
TINY = 1e-8
gamma = 0.98

# xavier initialization is another way to init weight randomly, but better
weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

# if we have the w's and b's saved, load it. Otherwise initialize it
if args.load_model:
    model = np.load(args.load_model)
    hw_init = tf.constant_initializer(model['hidden/weights'])
    hb_init = tf.constant_initializer(model['hidden/biases'])
    mw_init = tf.constant_initializer(model['mus/weights'])
    mb_init = tf.constant_initializer(model['mus/biases'])
    sw_init = tf.constant_initializer(model['sigmas/weights'])
    sb_init = tf.constant_initializer(model['sigmas/biases'])
else:
    hw_init = weights_init
    hb_init = relu_init
    mw_init = weights_init
    mb_init = relu_init
    sw_init = weights_init
    sb_init = relu_init

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 24
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.float32, shape=(None, output_units), name='y')

# 1 layer of hidden unit. Activation is ReLU
hidden = fully_connected(
    inputs=x,
    num_outputs=hidden_size,
    activation_fn=tf.nn.relu,
    weights_initializer=hw_init,
    weights_regularizer=None,
    biases_initializer=hb_init,
    scope='hidden')

# use last layer of neural network as phi(a, s) (the feature)
# mu = phi(s, a)^T dot theta
mus = fully_connected(
    inputs=hidden,
    num_outputs=output_units,
    activation_fn=tf.tanh,
    weights_initializer=mw_init,
    weights_regularizer=None,
    biases_initializer=mb_init,
    scope='mus')

# softplus is similar to ReLU. Activation function is g(x) = ln(1+e^x)
sigmas = tf.clip_by_value(fully_connected(
    inputs=hidden,
    num_outputs=output_units,
    activation_fn=tf.nn.softplus,
    weights_initializer=sw_init,
    weights_regularizer=None,
    biases_initializer=sb_init,
    scope='sigmas'),
    TINY, 5)

all_vars = tf.global_variables()

# use a Gaussian dist on pi because action is continuous
pi = tf.contrib.distributions.Normal(mus, sigmas, name='pi')
pi_sample = tf.tanh(pi.sample(), name='pi_sample')

# log probability of y given mu and sigma
log_pi = pi.log_prob(y, name='log_pi')

# Returns is a 1 x (T-1) array for float (rewards)
Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
for ep in range(16384):
    # reset the environment
    obs = env.reset()

    # generating all the states and actions and rewards
    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        env.render()
        # pi_sample is the list of randomly generated probablity
        # then we use pi_sample to generate the list of actions
        action = sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward * I)
        G += reward * I # G is the total discounted reward
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break
    # done generating

    if not args.load_model:
        # G_t = total - culmulative up to time t
        # set of all G_t's
        returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
        index = ep % MEMORY
        
        # ep_states contains all the state S_0 to S_T-1
        # ep_actions contains all the actions from A_0 to A_T-1
        # returns (ie reward) contains all the G_t's form t=0 to t=T
        _ = sess.run([train_op],
                    feed_dict={x:np.array(ep_states),
                                y:np.array(ep_actions),
                                Returns:returns })

    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                               mean_return))


    with tf.variable_scope("mus", reuse=True):
        print("incoming weights for the mu's from the first hidden unit:", sess.run(tf.get_variable("weights"))[0,:])


sess.close()
