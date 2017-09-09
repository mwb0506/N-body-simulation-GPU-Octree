import numpy as np
from numba import cuda, float32
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from time import clock
from math import sqrt


'''------------------------------------------------   Disclaimer   ------------------------------------------------'''


# # Only tested on a gtx 960m with compute capability 5.0
# # DWORD 'TdrDelay' and 'TdrLevel' at 8 and 0 respectively must be added to allow for larger amount of bodies in
# # HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
# #


'''------------------------------------------------   GPU function   ------------------------------------------------'''


@cuda.jit(device=True, inline=True)
def interaction(xi, yi, zi, axi, ayi, azi, xj, yj, zj, mj, eps, min_dist):
    """
    Models the gravitational interaction between two bodies
    :param xi: x-coordinate of body i
    :param yi: y-coordinate of body i
    :param zi: y-coordinate of body i
    :param axi: Acceleration of body i along the x-axis
    :param ayi: Acceleration of body i along the y-axis
    :param azi: Acceleration of body i along the z-axis
    :param xj: x-coordinate of body j
    :param yj: y-coordinate of body j
    :param zj: z-coordinate of body j
    :param mj: Mass of body j
    :param eps: Softening parameter
    :param min_dist: Minimal distance for acceleration calculation
    :return: Updated acceleration of body i along the three axes
    """
    rx = xj-xi  # Distance along the x-axis
    ry = yj-yi  # Distance along the y-axis
    rz = zj-zi  # Distance along the z-axis
    distance = sqrt(rx*rx+ry*ry+rz*rz)  # Distance between the bodies plus the softening parameter
    if distance > min_dist:  # If the bodies are far enough apart (at least not the same)
        scale = mj/(distance**3+eps)  # Scale for the distance vector (rx, ry, rz) for gravitation
    elif distance < 1.1*eps:  # If too close together (or the same particle)
        scale = 0  # Zero acceleration
    else:  # If the bodies are very close together
        scale = mj/(distance*min_dist**2+eps)  # Constant acceleration once scaled
    axi += rx*scale  # Add acceleration of body j to that of body i along the x-axis
    ayi += ry*scale  # Add acceleration of body j to that of body i along the y-axis
    azi += rz*scale  # Add acceleration of body j to that of body i along the z-axis
    return axi, ayi, azi


@cuda.jit(device=True, inline=True)
def block_acceleration(xi, yi, zi, axi, ayi, azi, bodies, eps, min_dist):
    """
    Executes interaction for each block in the grid
    :param xi: x-coordinate of body i
    :param yi: y-coordinate of body i
    :param zi: z-coordinate of body i
    :param axi: Acceleration of body i along the x-axis
    :param ayi: Acceleration of body i along the y-axis
    :param azi: Acceleration of body i along the z-axis
    :param bodies: Array of body vectors
    :param eps: Softening parameter
    :param min_dist: Minimal distance for acceleration calculation
    :return: Updated acceleration of body i along the three axes
    """
    for j in range(cuda.blockDim.x):  # For each body in this block
        xj = bodies[j, 0]  # x-coordinate of body j
        yj = bodies[j, 1]  # y-coordinate of body j
        zj = bodies[j, 2]  # z-coordinate of body j
        mj = bodies[j, 3]  # Mass of body j
        axi, ayi, azi = interaction(xi, yi, zi, axi, ayi, azi, xj, yj, zj, mj, eps, min_dist)  # Update acceleration
    return axi, ayi, azi


@cuda.jit('void(float32[:, :], float32, float32, float32)')
def total_acceleration(d_bodies, N, eps, min_dist):
    """
    Calculates the acceleration for each body
    :param d_bodies: Array of all the body vectors
    :param: N: Amount of bodies
    :param eps: Softening parameter
    :param min_dist: Minimal distance for acceleration calculation
    """
    block_dim = 32  # Manually set block size (can not be a parameter because of the shared array)
    sh_bodies = cuda.shared.array((block_dim, 4), float32)  # Create shared array for block_acceleration
    i = cuda.grid(1)  # Create grid
    xi = d_bodies[i, 0]  # x-coordinate of body i
    yi = d_bodies[i, 1]  # y-coordinate of body i
    zi = d_bodies[i, 2]  # z-coordinate of body i
    axi = 0.0  # Initialize acceleration of body i along the x-axis
    ayi = 0.0  # Initialize acceleration of body i along the y-axis
    azi = 0.0  # Initialize acceleration of body i along the z-axis
    for j in range(0, N, block_dim):  # For each block
        index = (j//block_dim)*cuda.blockDim.x+cuda.threadIdx.x  # Calculate index
        sh_index = cuda.threadIdx.x  # Identify the thread (body)
        sh_bodies[sh_index, 0] = d_bodies[index, 0]  # Add x-coordinate of body to shared array
        sh_bodies[sh_index, 1] = d_bodies[index, 1]  # Add y-coordinate of body to shared array
        sh_bodies[sh_index, 2] = d_bodies[index, 2]  # Add z-coordinate of body to shared array
        sh_bodies[sh_index, 3] = d_bodies[index, 9]  # Add mass of body to shared array
        cuda.syncthreads()  # Wait for the entire shared array to finish
        axi, ayi, azi = block_acceleration(xi, yi, zi, axi, ayi, azi, sh_bodies, eps, min_dist)  # Update acceleration
        cuda.syncthreads()  # Wait for all the accelerations to be updated
    d_bodies[i, 6] = axi  # Assign acceleration along the x-axis to body i
    d_bodies[i, 7] = ayi  # Assign acceleration along the y-axis to body i
    d_bodies[i, 8] = azi  # Assign acceleration along the z-axis to body i


@cuda.jit('void(float32[:, :], float32, float32)')
def leapfrog(d_bodies, delta_t, inv_step_size):
    """
    Executes the leapfrog integration method
    :param d_bodies: Array of all the body vectors
    :param delta_t: Time step duration
    :param inv_step_size: Full step (1) or half step (2)
    """
    i = cuda.grid(1)  # Create grid
    d_bodies[i, 3] += delta_t*d_bodies[i, 6]/inv_step_size  # Update velocity of body i along the x-axis
    d_bodies[i, 4] += delta_t*d_bodies[i, 7]/inv_step_size  # Update velocity of body i along the y-axis
    d_bodies[i, 5] += delta_t*d_bodies[i, 8]/inv_step_size  # Update velocity of body i along the z-axis
    d_bodies[i, 0] += delta_t*d_bodies[i, 3]  # Update x-coordinate of body i
    d_bodies[i, 1] += delta_t*d_bodies[i, 4]  # Update y-coordinate of body i
    d_bodies[i, 2] += delta_t*d_bodies[i, 5]  # Update z-coordinate of body i


'''-------------------------------------------------   Constants   -------------------------------------------------'''


G = 6.67*10**-11  # Gravitational constant
kpc = 3.08567758*10**19  # 1 Kilo parsec in meters
min_dist = 1  # Minimum distance between two bodies for calculating acceleration
eps = np.float32(1e-3)  # Softening parameter


'''----------------------------------------------------   Data   ----------------------------------------------------'''


'''
galaxy1 = np.zeros((1600, 10), dtype=np.float32)  # Allocate galaxy array
theta = np.repeat(np.arange(41)*2*np.pi/41, 39)  # Multiple sets of an array of angles
r = np.tile(np.arange(5, 200, 5), 41)  # Multiple sets of an array of radii
galaxy1[1:, 0] = r*np.cos(theta)  # Set x-coordinate from (r, theta)
galaxy1[1:, 1] = r*np.sin(theta)  # Set y-coordinate from (r, theta)
galaxy1[1:, 3] = -np.sqrt(G*1000000/r)*np.sin(theta)*1.2  # Perpendicular velocity .*sqrt(GM/r) in the x-direction
galaxy1[1:, 4] = np.sqrt(G*1000000/r)*np.cos(theta)*1.2  # Perpendicular velocity .*sqrt(GM/r) in the y-direction
galaxy1[1:, 9] = G  # Set masses of all 'planets' scaled by G
galaxy1[0, 9] = 1000000*G  # Set mass of the 'star' scaled by G
follow = 90  # State which particle to create a trail of
bodies = galaxy1  # Set galaxy1 as de body array

# Set colors of the bodies for the scatter plot
N = bodies.shape[0]  # Amount of bodies
color = np.zeros((N, 4), dtype=np.float32)  # Allocate color array
color[:, 3] = 0.9  # Set transparency to 90%
color[1:, 1] = 1  # Add color green to all but the center
color[0, 2] = 1  # Add color blue to center
'''

data = np.loadtxt("dubinskitab.txt")  # Load Milky Way & Andromeda data (m, x, y, z, vx, vy, vz)
# 16384 Gal. disk, 16384 And. disk, 8192  Gal. bulge, 8192  And. bulge, 16384 Gal. halo, 16384 And. halo
mod = 16  # Set amount 81920/mod amount of bodies

bodies = np.zeros((int(81920/mod), 10), dtype=np.float32)  # Initialize body array
bodies[:, :6] = data[0::mod, 1:]  # Copy positions and velocities of selected data
bodies[:, 9] = mod*data[0::mod, 0]  # Copy masses of selected data, balance total mass and scale with G
follow = 90  # State which particle to create a trail of


# Set color of the plot based on the Milky Way & Andromeda data
N = bodies.shape[0]  # Amount of bodies
color = np.zeros((N, 4), dtype=np.float32)  # Initialize color array for the plot
color[:, 3] = 0.3  # Set transparency to 30%
color[:int(16384/mod), 2] = 1  # Set Milky Way core to blue
color[int(16384/mod):int(32768/mod), 0] = 1  # Set Andromeda core to red
color[int(32768/mod):int(40960/mod), 2] = 0.8  # Set Milky Way bulge to lighter blue
color[int(409604/mod):int(49152/mod), 0] = 0.8  # Set Andromeda bulge to lighter red
color[int(49152/mod):int(65536/mod), 2] = 0.6  # Set Milky way halo to even lighter blue
color[int(65536/mod):, 0] = 0.6  # Set Andromeda halo to even lighter red


d_bodies = cuda.to_device(bodies)  # Copy the array of body vectors to the gpu


'''-------------------------------------------------   Parameters   -------------------------------------------------'''


# Set block and grid dimensions for the GPU
blockdim = 32  # Amount of threads per block
griddim = int(np.ceil(N/blockdim))  # Amount of blocks

# Initialize GPU functions with the given grid and block dimensions
leapfrog = leapfrog.configure(griddim, blockdim)  # Configure leapfrog function
total_acceleration = total_acceleration.configure(griddim, blockdim)  # Configure total_acceleration function


'''------------------------------------------------   3D Animation   ------------------------------------------------'''


app = QtGui.QApplication([])  # Initialize application

# Initialize window for the scatter plots
w = gl.GLViewWidget()  # Initialize opengl widget
w.opts['distance'] = 12500  # Set viewing distance to the figure
w.show()  # Show the figure
w.setWindowTitle('N-body simulation')  # Set title of the window
w.setGeometry(960, 35, 960, 995)  # Set window to envelop right side of the screen

# Scatter plot of all the bodies
sp = gl.GLScatterPlotItem(pos=bodies[:, :3], color=color, size=7)  # Set initial frame
sp.scale(20, 20, 20)  # Scale the plot to match the grids
sp.translate(-10, -10, -10)  # Translate the plot to match the grids
sp.rotate(80, 0, 0, 1)
w.addItem(sp)  # Add plot to figure

# Initialize arrays for the scatter plot trail
bodies2 = np.zeros((1000000, 3))  # Allocate memory for the body position over time
bodies2[0, :] = bodies[follow, :3]  # Set the location at the first frame
color2 = np.zeros((1000000, 4))  # Allocate memory for the color array
color2[:, 0] = 1  # Set color array to color of followed particle
color2[:, 3] = 0.9  # Set color array to color of followed particle

# Scatter plot of the trail of a single body
sp2 = gl.GLScatterPlotItem(pos=bodies2[0, :], color=color2[0, :], size=5, glOptions='additive')  # Set initial frame
sp2.scale(20, 20, 20)  # Scale the plot to match the grids
sp2.translate(-10, -10, -10)  # Translate the plot to match the grids
w.addItem(sp2)  # Add plot to figure

# Initialize window for the scrolling plots
win = pg.GraphicsWindow()  # Initialize window for plotting
win.setGeometry(0, 35, 960, 995)  # Set window to envelop left side of the screen
p1 = win.addPlot(row=1, col=0)  # Add top plot
p2 = win.addPlot(row=2, col=0)  # Add bottom plot

# Scrolling plots of the time steps
dts = np.zeros(1000000, dtype=np.float32)  # Allocate array for dt's
curve = p1.plot(dts[:500])  # Set initial frame top plot
curve2 = p2.plot(dts[:2])  # Set initial frame bottom plot

frame = 0  # Frame index


def update():
    """
    Updates the plot
    """
    global sp, d_bodies, N, frame, ts, color, dts, bodies2, color2, dt
    frame += 1  # Update frame index

    # Update the positions
    for amount in range(10):  # Calculates amount of frames per one display of a frame
        total_acceleration(d_bodies, N, eps, min_dist)  # Calculate the accelerations
        d_bodies.copy_to_host(bodies)  # Copy the updated array of body vectors back to RAM (auto syncs threads)
        dt = np.clip(1/(np.max(np.abs(bodies[:, 6:9]))+0.001), 0, 200)  # Set time step to max velocity, acceleration
        if frame == 1:  # Half step of the velocity
            ts = clock()  # Starting time
            #dts[499] = dt  # Replace initial value needed for initializing the plot
            leapfrog(d_bodies, dt, 2)  # Apply the leapfrog integration method
        else:  # Full step of the velocity
            leapfrog(d_bodies, dt, 1)  # Apply the leapfrog integration method

    # Top plot
    dts[499+frame] = dt  # Save value of current frame
    curve.setData(dts[frame:500+frame])  # Set data as last 500 frames
    curve.setPos(frame-499, 0)  # Scroll x-axis

    # Bottom plot
    curve2.setData(dts[499:500+frame])  # Set data as all elapsed frames

    # Plot bodies
    d_bodies.copy_to_host(bodies)  # Copy the updated array of body vectors back to RAM (auto syncs threads)
    sp.setData(pos=bodies[:, :3])  # Update the plot data and colors

    # Plot Trail
    bodies2[frame, :] = bodies[follow, :3]  # Save value of current frame
    sp2.setData(pos=bodies2[:frame+1, :], color=color2[:frame+1, :])  # Update the plot data and colors

    # Print FPS
    if frame % 10 == 0:  # Display fps every 10 frames
        print(10/(clock()-ts))  # Display fps
        ts = clock()  # Reset starting time

# Update the figure
timer = QtCore.QTimer()  # Initialize timer
timer.timeout.connect(update)  # Connect the timer to the update function
timer.start(0)  # Start timer
QtGui.QApplication.instance().exec_()  # Run the figure
