import numpy as np
from numpy import linalg as np_linalg
from time import clock
import time
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from scipy.spatial import distance as sp_dist

num_of_nodes = 0


# Octree class
class Octree:
    """Octree object controlling the Barnes-Hut iterative process

    Attributes:
        bodies (ndarray): List of bodies in the system
        root (Node): Root node of the Octree
    """

    def __init__(self, bodies):
        """Initialises the octree with given region

        Args:
            bodies (ndarray): List of bodies in the system
        """

        self.bodies = np.array(bodies)

    def create(self):
        """Creates the octree"""
        global num_of_nodes
        num_of_nodes = 1
        cube = np.append(np.amin(self.bodies[:, :3], axis=0), np.amax(self.bodies[:, :3], axis=0))  # Bounding box for the bodies
        self.root = Node(cube)  # Create the root node
        self.root.add_body(self.bodies)  # Add the bodies to the root node

    def update(self, dt):
        """Updates the locations of the bodies in the octree

        Args:
            dt (float): Time-step size
        """
        self.root.calc_force(self.bodies, np.arange(self.bodies.shape[0]), 1000)  # Calculate accelerations

        self.bodies[:, 3:6] += dt * self.bodies[:, 6:9]  # Update velocity
        self.bodies[:, 0:3] += dt * self.bodies[:, 3:6]  # Update position
        self.bodies[:, 6:9] = 0  # Clear acceleration


class Node:
    """A node in the Barnes-Hut octree

    Attributes:
        center_of_mass (array): Center of mass for all bodies in this node's region
        children (list): List of child nodes
        cube (list): Region corresponding to this node
        type (int): Type of the node (0 for empty, 1 for external and 2 for internal)
    """

    empty = 0
    external = 1
    internal = 2

    def __init__(self, cube):
        """Initialises a node

        Args:
            cube (list): Region corresponding to this node
        """

        self.cube = cube
        self.children = [None, None, None, None, None, None, None, None]
        self.center_of_mass = np.zeros(4, dtype=np.float32)
        self.type = Node.empty

    def child_cube(self, index):
        """Finds the region corresponding to the child node with given index

        Args:
            index (int): Index of the child object

        Returns:
            list: Region contained by the child object
        """

        cube = np.zeros(6)
        if index >= 4:
            # Upper half in z
            cube[2] = (self.cube[2] + self.cube[5])/2  # Midpoint
            cube[5] = self.cube[5]
            index -= 4  # Remove z component from index

        else:
            # Lower half in z
            cube[2] = self.cube[2]
            cube[5] = (self.cube[2] + self.cube[5])/2  # Midpoint

        if index >= 2:
            # Upper half in y
            cube[1] = (self.cube[1] + self.cube[4])/2  # Midpoint
            cube[4] = self.cube[4]
            index -= 2  # Remove y component from index

        else:
            # Lower half in y
            cube[1] = self.cube[1]
            cube[4] = (self.cube[1] + self.cube[4])/2  # Midpoint

        if index >= 1:
            # Upper half in x
            cube[0] = (self.cube[0] + self.cube[3])/2  # Midpoint
            cube[3] = self.cube[3]

        else:
            # Lower half in x
            cube[0] = self.cube[0]
            cube[3] = (self.cube[0] + self.cube[3])/2  # Midpoint

        return cube

    def add_body(self, bodies):
        """Adds a body to this node

        Args:
            bodies (ndarray): Two-dimensional array containing the bodies of the system.
                              It is assumed that bodies[i, :3] = (x,y,z) and bodies[i, 9] = mmass
        """
        global num_of_nodes

        if bodies.shape[0] == 1:  # Only one body to add to this node
            self.center_of_mass[:3] = bodies[0, :3]  # Set center of mass position equal to body position
            self.center_of_mass[3] = bodies[0, 9]  # Set total mass equal to body mass
            self.type = Node.external  # External node

        else:  # If more than one body
            # Calculate weighted average of body positions and total mass for center of mass
            (self.center_of_mass[:3], totalmass) = np.average(bodies[:, :3], axis=0,
                                                              weights=bodies[:, 9], returned=True)
            self.center_of_mass[3] = totalmass[0]
            self.type = Node.internal  # Internal node

            # Find indices of children to pass bodies
            child_index = np.zeros(bodies.shape[0])
            child_index[bodies[:, 0] >= ((self.cube[3]+self.cube[0])/2)] += 1
            child_index[bodies[:, 1] >= ((self.cube[4]+self.cube[1])/2)] += 2
            child_index[bodies[:, 2] >= ((self.cube[5]+self.cube[2])/2)] += 4

            for i in range(8):  # For all children
                to_pass = bodies[child_index == i, :]  # Define bodies to pass to child
                if to_pass.shape[0] != 0:  # If more than one body to pass
                    num_of_nodes += 1
                    self.children[i] = Node(self.child_cube(i))  # Initialise child
                    self.children[i].add_body(to_pass)  # Add bodies to child

    def calc_force(self, bodyList, indices, theta):
        """Calculates the force on a body

        Args:
            bodyList (ndarray): Two-dimensional array containing the bodies of the system.
                                Assumed data: (x, y, z, vx, vy, vz, ax, ay, az, m)
            indices (list): List of indices to change in this node
            theta (float): Pruning threshold
        """
        s = np.max(self.cube[3:] - self.cube[:3])  # Metric for size of cube
        d = sp_dist.cdist(self.center_of_mass[:3].reshape((1, 3)), bodyList[indices, :3])  # Distance between bodies and center
        a = np.zeros((indices.size, 3), dtype=np.float32)

        # Calculate force here if the cluster is too small OR it is an external node.
        # The condition d > 1e-3 is put to avoid a body interacting with itself
        calc_cond = ((s < d[0, :]*theta) | self.type != Node.internal) & (d[0, :] > 1e-3)

        # Calculate acceleration
        a[calc_cond, :] = self.center_of_mass[3]*(self.center_of_mass[:3] -
                                                  bodyList[indices[calc_cond], :3])/((d[0, calc_cond]**3 + 0.001)[:, None])

        bodyList[indices, 6:9] += a  # Update

        pass_index = indices[s >= d[0, :]*theta]  # Indices to be evaluated in child nodes
        for child in self.children:
            if child is not None:
                child.calc_force(bodyList, indices[s >= d[0, :]*theta], theta)  # Calculate force in child node

    def __del__(self):
        """Deletes the node"""
        self.center_of_mass = None
        self.children = None
        self.cube = None

PERFORMANCE_TEST = 0  # Set to 1 for a performance test
STABILITY_TEST = 0  # Set to 1 for a stability test
ANIMATION = 1  # Set to 1 for real-time evaluation
dt = 0.003  # Timestep

if PERFORMANCE_TEST:
    N = 1280  # Base size
    steps = 7  # Number of times the size must be doubled
    c_times = np.zeros(steps)
    c_times_exp = np.zeros(steps)
    u_times = np.zeros(steps)
    u_times_exp = np.zeros(steps)
    t_times = np.zeros(steps)
    t_times_exp = np.zeros(steps)

    for i in range(steps):
        N_test = N * 2 ** i  # Size to test performance

        # Initialise normal distribution for body position and velocity
        bodies = np.zeros((N_test, 10), dtype=np.float32)
        for j in range(N_test):
            # bodies[j, :6] = np.random.uniform(low=-1, high=1, size=6)
            bodies[j, :6] = np.random.normal(loc=0, scale=0.3, size=6)
            bodies[j, -1] = 0.1

        Oct = Octree(bodies)

        create_time = 0
        update_time = 0

        N_it = 10  # Number of iterations per size

        for k in range(N_it):  # Test performance
            start = time.clock()
            Oct.create()
            end = time.clock()
            create_time += end - start

            start = time.clock()
            Oct.update(dt)
            end = time.clock()
            update_time += end - start

        # Save times
        c_times[i] = create_time/N_it
        u_times[i] = update_time/N_it
        t_times[i] = c_times[i] + u_times[i]

        # Update expected times (linear or N log N)
        if i == 0:
            c_times_exp[i] = c_times[i]
            u_times_exp[i] = u_times[i]
            t_times_exp[i] = t_times[i]
        else:
            c_times_exp[i] = 2*c_times_exp[i-1]  # *(1+np.log10(2)/np.log10(N/2))
            u_times_exp[i] = 2*u_times_exp[i-1]*(1+np.log10(2)/np.log10(N_test/2))
            t_times_exp[i] = 2*t_times_exp[i-1]*(1+np.log10(2)/np.log10(N_test/2))

        print("N = %5d done! Number of nodes: %7d" % (N_test, num_of_nodes))

    # Print results
    print('\n\n\n\n\n')
    print("TREE CREATION TIMES")
    print("     N     |   Times   |  Expected ")
    print("-----------------------------------")
    for i in range(steps):
        print(' %9d | %9.6f | %9.6f ' % (N * 2 ** i, c_times[i], c_times_exp[i]))

    print('\n\n\n\n\n')
    print("TREE UPDATE TIMES")
    print("     N     |   Times   |  Expected ")
    print("-----------------------------------")
    for i in range(steps):
        print(' %9d | %9.6f | %9.6f ' % (N * 2 ** i, u_times[i], u_times_exp[i]))

    print('\n\n\n\n\n')
    print("TREE TOTAL FRAMERATE")
    print("     N     | Framerate |  Expected ")
    print("-----------------------------------")
    for i in range(steps):
        print(' %9d | %9.4f | %9.4f ' % (N * 2 ** i, 1/t_times[i], 1/t_times_exp[i]))

if STABILITY_TEST:
    N = 320  # Size to test

    # Initialise bodies
    bodies = np.zeros((N, 10), dtype=np.float32)
    for j in range(N):
        # bodies[j, :6] = np.random.uniform(low=-1, high=1, size=6)
        bodies[j, :6] = np.random.normal(loc=0, scale=0.5, size=6)
        bodies[j, -1] = 0.1

    Oct = Octree(bodies)

    create_time = 0
    update_time = 0

    N_it = 100  # Number of time-steps for forward and backward iterations
    print("Starting forward iteration...")

    for k in range(N_it):  # Forward iterations
        Oct.create()
        Oct.update(dt)

    print("\nStarting backward iteration...")

    for i in range(N_it):  # Backward iteration
        Oct.create()
        Oct.update(-dt)

    # Print results
    print("\nMax Difference:", np.max(Oct.bodies[:, :3]-bodies[:, :3]))
    print("Norm Difference:", np_linalg.norm(Oct.bodies[:, :3]-bodies[:, :3]))


'''------------------------------------------------   3D Animation   ------------------------------------------------'''

if ANIMATION:
    N = 96
    # Initialise bodies
    G = 1
    # bodies = np.zeros((N, 10), dtype=np.float32)

    # for i in range(N):
    #     bodies[i, :3] = np.random.rand(3)
    #     bodies[:, -1] = 0.1
    bodies = np.zeros((96, 10), dtype=np.float32)  # Initialize body array
    theta = np.repeat(np.arange(19)*2*np.pi/19, 5)
    r = np.tile(np.arange(40, 80, 8), 19)
    bodies[1:, 0] = r*np.cos(theta)
    bodies[1:, 1] = r*np.sin(theta)
    bodies[1:, 3] = np.sqrt(G*1000000/r)*np.sin(theta)*1.2
    bodies[1:, 4] = -np.sqrt(G*1000000/r)*np.cos(theta)*1.2
    bodies[1:, 9] = G
    bodies[0, 9] = 1000000*G

    Oct = Octree(bodies)

    # Initialize
    app = QtGui.QApplication([])  # Initialize figure
    w = gl.GLViewWidget()  # Initialize opengl widget
    w.opts['distance'] = 200  # Set viewing distance to the figure
    w.show()  # Show the figure
    w.setWindowTitle('N-body simulation')  # Set title of the window
    w.showFullScreen()  # Show fullscreen (alt+f4 to quit)

    color = np.zeros((N, 4), dtype=np.float32)  # Initialize color array for the plot
    color[:, 3] = 0.9
    color[-1, 0] = 1
    color[1:-1, 1] = 1
    color[0, 2] = 1

    # Set color of the plot based on the Milky Way & Andromeda data
    # color = np.zeros((N, 4), dtype=np.float32)  # Initialize color array for the plot
    # color[:, 3] = 0.3  # Set transparency to 30%
    # color[:int(16384/mod), 2] = 1  # Set Milky Way core to blue
    # color[int(16384/mod):int(32768/mod), 0] = 1  # Set Andromeda core to red
    # color[int(32768/mod):int(40960/mod), 2] = 0.9  # Set Milky Way bulge to lighter blue
    # color[int(409604/mod):int(49152/mod), 0] = 0.9  # Set Andromeda bulge to lighter red
    # color[int(49152/mod):int(65536/mod), 2] = 0.8  # Set Milky way halo to even lighter blue
    # color[int(65536/mod):, 0] = 0.8  # Set Andromeda halo to even lighter red

    sp = gl.GLScatterPlotItem(pos=bodies[:, :3], color=color, size=25)
    sp.scale(20, 20, 20)  # Scale the plot to match the grids
    sp.translate(-10, -10, -10)  # Translate the plot to match the grids
    w.addItem(sp)  # Add plot to figure

    bodies2 = np.zeros((1000000, 3))
    bodies2[0, :] = bodies[-1, :3]
    color2 = np.zeros((1000000, 4))
    color2[0, :] = color[-1, :]
    sp2 = gl.GLScatterPlotItem(pos=bodies2, color=color2, size=5, glOptions='additive')
    sp2.scale(20, 20, 20)  # Scale the plot to match the grids
    sp2.translate(-10, -10, -10)  # Translate the plot to match the grids
    w.addItem(sp2)  # Add plot to figure

    frame = 0  # Frame index

    def update():
        """
        Updates the plot
        """
        global sp, d_bodies, N, frame, ts, color, dt, dts, curve, bodies2, color2
        frame += 1  # Update frame index
        if frame == 1:  # Half step of the velocity
            ts = clock()  # Starting time
            Oct.create()
            Oct.update(dt)
        else:  # Full step of the velocity
            Oct.create()
            Oct.update(dt)

        sp.setData(pos=Oct.bodies[:, 0:3])  # Update the plot data and colors
        bodies2[frame, :] = Oct.bodies[-1, :3]
        color2[frame, :] = color[-1, :]
        sp2.setData(pos=bodies2[:frame+1, :], color=color2[:frame+1, :])  # Update the plot data and colors
        if frame % 10 == 0:  # Display fps every 10 frames
            print(10/(clock()-ts))  # Display fps
            # print(Oct.root.center_of_mass)
            ts = clock()  # Begin new starting time

    # Update the figure
    timer = QtCore.QTimer()  # Initialize timer
    timer.timeout.connect(update)  # Connect the timer to the update function
    timer.start(1)  # Start timer
    QtGui.QApplication.instance().exec_()  # Run the figure
