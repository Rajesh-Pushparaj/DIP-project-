import numpy as np
from casadi import *
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import scipy.io as sio
import os.path as osp
import pdb

""" This class allows you to simulate the system and obtain the animation """
class pendulum_simulator:

    def __init__(self,x0):
        """ Initialize pendulum with initial value """
        self.X      = [np.reshape(x0,(-1,1))] # collection of all states
        self.U      = [] # collection of all inputs

        # parameters
        self.g      = 9.8067 # m/s^2, gravital acceleration
        self.m0     = 0.6    # kg, mass of the cart
        self.m1     = 0.2    # kg, mass of the first rod
        self.m2     = 0.2    # kg, mass of the second rod
        self.L1     = 0.5    # m, length of the first rod
        self.L2     = 0.5    # m, Length of the second rod

        # states
        x_cart       = SX.sym("x_cart", 1, 1)       # position cart
        theta_1      = SX.sym("theta_1", 1, 1)      # angle rod 1
        theta_2      = SX.sym("theta_2", 1, 1)      # angle rod 2
        dx_cart      = SX.sym("dx_cart", 1, 1)      # velocity cart
        dtheta_1     = SX.sym("dtheta_1", 1, 1)     # angular velocity rod 1
        dtheta_2     = SX.sym("dtheta_2", 1, 1)     # angular velocity rod 2
        self.x       = vertcat(x_cart,theta_1,theta_2,dx_cart,dtheta_1,dtheta_2)

        # algebraic states
        ddx_cart     = SX.sym("ddx_cart", 1, 1)
        ddtheta_1    = SX.sym("ddtheta_1", 1, 1)
        ddtheta_2    = SX.sym("ddtheta_2", 1, 1)
        self.z       = vertcat(ddx_cart,ddtheta_1,ddtheta_2)

        # input
        F            = SX.sym("F", 1, 1)            # Force acting on cart
        self.u       = vertcat(F)

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
        self.x_dot = vertcat(x_cart_dot,theta_1_dot,theta_2_dot,dx_cart_dot,dtheta_1_dot,dtheta_2_dot)

        # algebraic equations
        alg_ddx_cart = h1*ddx_cart + h2*ddtheta_1*cos(theta_1) + h3*ddtheta_2*cos(theta_2) - (h2*dtheta_1**2*sin(theta_1) + h3*dtheta_2**2*sin(theta_2) + F)
        alg_ddtheta_1 = h2*cos(theta_1)*ddx_cart + h4*ddtheta_1 + h5*cos(theta_1-theta_2)*ddtheta_2 - (h7*sin(theta_1) - h5*dtheta_2**2*sin(theta_1-theta_2))
        alg_ddtheta_2 = h3*cos(theta_2)*ddx_cart + h5*cos(theta_1-theta_2)*ddtheta_1 + h6*ddtheta_2 - (h5*dtheta_1**2*sin(theta_1-theta_2) + h8*sin(theta_2))
        self.alg = vertcat(alg_ddx_cart,alg_ddtheta_1,alg_ddtheta_2)

        # initialize simulator
        dae = {'x':self.x, 'z': self.z, 'p':self.u, 'ode':self.x_dot, 'alg':self.alg}
        opts = {'abstol': 1e-10, 'reltol': 1e-10, 'tf': 0.04}
        self.simulator = integrator("simulator", "idas", dae,  opts)

    def simulate(self,u):
        # simulate
        u = np.reshape(u,(-1,1))
        result_int = self.simulator(x0 = self.X[-1], p = u)
        # update lists
        self.U.append(u)
        self.X.append(np.reshape(result_int['xf'],(-1,1)))

        return np.reshape(result_int['xf'],(-1,1))

    def pendulum_bars(self,t_ind):

        x = self.X[t_ind].flatten()
        # Get the x,y coordinates of the two bars for the given state x.
        line_1_x = np.array([
        x[0],
        x[0]+self.L1*np.sin(x[1])
        ])

        line_1_y = np.array([
            0,
            self.L1*np.cos(x[1])
        ])

        line_2_x = np.array([
            line_1_x[1],
            line_1_x[1] + self.L2*np.sin(x[2])
        ])

        line_2_y = np.array([
            line_1_y[1],
            line_1_y[1] + self.L2*np.cos(x[2])
        ])

        line_1 = np.stack((line_1_x, line_1_y))
        line_2 = np.stack((line_2_x, line_2_y))

        return line_1, line_2

    def update(self,t_ind):
        """ Update the position of the two bars representing the pendulum """
        line1, line2 = self.pendulum_bars(t_ind)
        self.bar1[0].set_data(line1[0],line1[1])
        self.bar2[0].set_data(line2[0],line2[1])
        # adjust the window
        x_vals = np.hstack([line1[0],line2[0]])
        if any(x_vals<(self.cur_left+2.0)):
            self.cur_left = np.min(x_vals)-2.0
            self.cur_right = self.cur_left + 6.0
        elif any(x_vals>(self.cur_right-2.0)):
            self.cur_right = np.max(x_vals) + 2.0
            self.cur_left = self.cur_right - 6.0
        self.ax.set_xlim([self.cur_left,self.cur_right])

    def export_gif(self):
        # initialize figure
        fig = plt.figure(figsize=(16,9),tight_layout=True)
        self.ax = fig.add_subplot(1,1,1)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylim([-1.4,1.4])
        self.cur_left = -3.0
        self.cur_right = 3.0
        self.ax.set_xlim([self.cur_left,self.cur_right])
        self.ax.axhline(0,color='black')
        # The two bar objects represent the double pendulum
        self.bar1 = self.ax.plot([],[], '-o', linewidth=5, markersize=10)
        self.bar2 = self.ax.plot([],[], '-o', linewidth=5, markersize=10)
        self.ax.set_aspect('equal')
        anim = FuncAnimation(fig, self.update, frames=len(self.X), repeat=False)
        gif_writer = ImageMagickWriter(fps=20)
        anim.save('anim_dip.gif', writer=gif_writer)

    def export_data(self,filename):

        exp_dict = {'X':self.X,'F':self.U}
        sio.savemat(filename + '.mat',exp_dict)
