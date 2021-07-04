import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class dipEnv(gym.Env):

#metadata = {
        #'render.modes': ['human', 'rgb_array'],
       # 'video.frames_per_second': 50
   # }
   
    def __init__(self):
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.force_mag = 4.0
        
        # parameters
        self.g      = 9.8067 # m/s^2, gravital acceleration
        self.m0     = 0.6    # kg, mass of the cart
        self.m1     = 0.2    # kg, mass of the first rod
        self.m2     = 0.2    # kg, mass of the second rod
        self.L1     = 0.5    # m, length of the first rod
        self.L2     = 0.5    # m, Length of the second rod
        
                
          # Angle at which to fail the episode
        self.theta_1_threshold_radians = 12 * 2 * math.pi / 360
        self.theta_2_threshold_radians = 12 * 2 * math.pi / 360
        self.x_cart_threshold = 0.5
        
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds.
        high = np.array([self.x_cart_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_1_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         self.theta_2_threshold_radians * 2,
                         np.finfo(np.float32).max],
                         dtype=np.float32)

        self.action_space = spaces.Box(low = -self.force_mag, 
            high = self.force_mag,
            shape = (1,),
            dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
        
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        x_cart, theta_1, theta_2, dx_cart, dtheta_1, dtheta_2 = self.state
        F = min(max(action[0], -self.force_mag),self.force_mag) # Force acting on cart
        
        
        ddx_cart     = np.array([])
        ddtheta_1    = np.array([])
        ddtheta_2    = np.array([])

        # auxiliary terms
        l1 = self.L1/2 # m,
        l2 = self.L2/2 # m,
        J1 = (self.m1 * l1**2) / 3   # Inertia
        J2 = (self.m2 * l2**2) / 3   # Inertia

        h1           = self.m0 + self.m1 + self.m2
        h2           = self.m1*l1 + self.m2*self.L1
        h3           = self.m2*l2
        h4           = self.m1*l1**2 + self.m2*self.L1**2 + J1
        h5           = self.m2*l2*self.L1
        h6           = self.m2*l2**2 + J2
        h7           = (self.m1*l1 + self.m2*self.L1) * self.g
        h8           = self.m2*l2*self.g
        
        # ODEs
        x_cart_dot = dx_cart
        theta_1_dot = dtheta_1
        theta_2_dot = dtheta_2
        dx_cart_dot = ddx_cart
        dtheta_1_dot = ddtheta_1
        dtheta_2_dot = ddtheta_2

        # algebraic equations
        alg_ddx_cart = h1*ddx_cart + h2*ddtheta_1*math.cos(theta_1) + h3*ddtheta_2*math.cos(theta_2) - (h2*dtheta_1**2*math.sin(theta_1) + h3*dtheta_2**2*math.sin(theta_2) + F)
        alg_ddtheta_1 = h2*math.cos(theta_1)*ddx_cart + h4*ddtheta_1 + h5*math.cos(theta_1-theta_2)*ddtheta_2 - (h7*math.sin(theta_1) - h5*dtheta_2**2*math.sin(theta_1-theta_2))
        alg_ddtheta_2 = h3*math.cos(theta_2)*ddx_cart + h5*math.cos(theta_1-theta_2)*ddtheta_1 + h6*ddtheta_2 - (h5*dtheta_1**2*math.sin(theta_1-theta_2) + h8*math.sin(theta_2))

                
        if self.kinematics_integrator == 'euler':
            x_cart = x_cart + self.tau * dx_cart
            dx_cart = dx_cart + self.tau * alg_ddx_cart
            theta_1 = theta_1 + self.tau * dtheta_1
            dtheta_1 = dtheta_1 + self.tau * alg_ddtheta_1
            theta_2 = theta_2 + self.tau * dtheta_2
            dtheta_2 = dtheta_2 + self.tau * alg_ddtheta_2
        else:  # semi-implicit euler
            dx_cart = dx_cart + self.tau * alg_ddx_cart
            x_cart = x_cart + self.tau * dx_cart
            dtheta_1 = dtheta_1 + self.tau * alg_ddtheta_1
            theta_1 = theta_1 + self.tau * dtheta_1
            dtheta_2 = dtheta_2 + self.tau * alg_ddtheta_2
            theta_2 = theta_2 + self.tau * dtheta_2
            
        self.state = (x_cart, theta_1, theta_2, x_cart_dot, theta_1_dot, theta_2_dot)
        
                      
        done = bool(
            x_cart < -self.x_cart_threshold
            or x_cart > self.x_cart_threshold
            or theta_1 < -self.theta_1_threshold_radians
            or theta_1 > self.theta_1_threshold_radians
            or theta_2 < -self.theta_2_threshold_radians
            or theta_2 > self.theta_2_threshold_radians
        )
        
        
        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1
        
        '''if not done:
            reward = -1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0'''
            
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.steps_beyond_done = None
        return np.array(self.state)
        