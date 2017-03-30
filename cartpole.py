"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *
import sys

logger = logging.getLogger(__name__)

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        

""" This is where part 2 coding starts """
env = gym.make('CartPole-v0')
cp = CartPoleEnv()

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
cp._seed(RNG_SEED)
np.random.seed(RNG_SEED)

alpha = 0.0001
gamma = 0.99

# xavier initialization is another way to init weight randomly, but better
weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

w_init = weights_init
b_init = relu_init

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
x = tf.placeholder(tf.float32, shape=(None, input_shape), name='x')
y = tf.placeholder(tf.int32, shape=(None, 1), name='y')

# 1 layer oin neural network. Activation is ReLU
output = fully_connected(
    inputs=x,
    num_outputs=output_units,
    activation_fn=tf.nn.relu,
    weights_initializer=w_init,
    weights_regularizer=None,
    biases_initializer=b_init,
    scope='output')

# adde a softmax
soft_max_full = tf.nn.softmax(output)
# grab the first col of softmax
soft_max = tf.reshape(soft_max_full[:,0], [tf.shape(soft_max_full)[0], 1])

# use Bernoulli distribution
pi = tf.contrib.distributions.Bernoulli(soft_max, name='pi')
pi_sample = pi.sample()

# log probability of y
log_pi = pi.log_prob(y, name='log_pi')

# Returns is a 1 x (T-1) array for float (rewards)
Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
cost = -1.0 * Returns * log_pi
train_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
for ep in range(10001):
    # reset the environment
    obs = cp._reset()

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
        #cp._render()
        
        # pi_sample is the list of randomly generated probablity
        # then we use pi_sample to generate the list of actions
        action = sess.run(pi_sample, feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = cp._step(action)
        ep_rewards.append(reward * I)
        G += reward * I # G is the total discounted reward
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break
    # done generating

    # G_t = total - culmulative up to time t
    # set of all G_t's
    returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
    
    # ep_states contains all the state S_0 to S_T-1
    # ep_actions contains all the actions from A_0 to A_T-1
    # returns (ie reward) contains all the G_t's form t=0 to t=T
    _ = sess.run([train_op],
                feed_dict={x:np.array(ep_states),
                            y:np.reshape(np.array(ep_actions), (len(ep_actions), 1)),
                            Returns:returns })

    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    
    
    if (ep % 500 == 0):
        print("Episode {} finished after {} steps with return {}".format(ep, t, G))
        print("Mean return over the last {} episodes is {}".format(MEMORY, mean_return))
        print("Cost: ", sess.run(tf.reduce_sum(cost), feed_dict={x:np.array(ep_states),y:np.reshape(np.array(ep_actions), (len(ep_actions), 1)), Returns:returns }))
    
    
        # with tf.variable_scope("output", reuse=True):
        #     print("incoming weights for the output", sess.run(tf.get_variable("weights"))[0,:])
        # print()

"""
sess.close()
"""