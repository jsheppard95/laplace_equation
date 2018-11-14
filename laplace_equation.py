#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:11:29 2017

@author: Jackson Sheppard
"""

# Phys 110A
# HW 3
# Extra Credit: The Generalized Pringle

# Program uses method of relaxation to approximate 2D potential V(x,y) in
# square region -1 < x < 1, -1 < y < 1
# No charges inside region
# edges held at V_edge = (sin(3*arctan(y/x)))^2
# Plots results as a 3D surface plot V(x,y) vs. x,y

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

MAX_ITER = 10000 # max number of iterations
NPTS = 350 # number of points between edges of grid
XMIN = -1
XMAX = 1
XLEN = np.abs(XMAX - XMIN)
YMIN = -1
YMAX = 1
YLEN = np.abs(YMAX - YMIN)

t0 = time.perf_counter()
# Create grid, set each point to 0:
grid = np.zeros((NPTS, NPTS))

# Define Boundary condition: V_edge
def func(x, y):
    return (np.sin(3*np.arctan(y/x)))**2

def tx(x, size):
    """This function will horizontal indeces in the grid array (i,j) to 
    traditional (x,y) cartesian coordinates"""
    x -= size/XLEN
    x *= (XLEN/size)
    return x
    
def ty(y, size):
    """similar function to tx but for vertical indeces"""
    y -= size/YLEN
    y *= (YLEN/size)
    y = -y
    return y
    
def load_boundary(a):
    """This function will load the boundary conditions into the voltage array
    corresponding to the grid. Called after each iteration to reset boundary
    conditions
    a: array corresponding to grid
    """
    # Load top border
    top_row = 0
    top_coord = ty(top_row, NPTS)
    for i in np.arange(NPTS):
        x = tx(i, NPTS) # transform index to its x coordinate
        a[top_row][i] = func(x, top_coord)
    
    # Load botom border
    bottom_row = NPTS - 1
    bottom_coord = ty(bottom_row, NPTS)
    for i in np.arange(NPTS):
        x = tx(i, NPTS)
        a[bottom_row][i] = func(x, bottom_coord)
    
    # Load left border
    left_col = 0
    left_coord = tx(left_col, NPTS)
    for i in np.arange(NPTS):
        y = ty(i, NPTS)
        a[i][left_col] = func(left_coord, y)
    
    # Load Right border
    right_col = NPTS - 1
    right_coord = tx(right_col, NPTS)
    for i in np.arange(NPTS):
        y = ty(i, NPTS)
        a[i][right_col] = func(right_coord, y)

print("Loading Boundary...")        
load_boundary(grid)


# Now begin relaxation algorithm
# Iterate through interior of grid, setting each point to average of four
# points surrounding it
next_grid = np.zeros((NPTS, NPTS)) # copy of grid to adjust and compare
np.copyto(next_grid, grid) # keeps arrays disconnected

# Continue averaging until change in solution is negligible

print("Solving Laplace's Equation by Relaxation...")
curr_iter = 0
while curr_iter < MAX_ITER:
    # Average each point in grid with 4 surrounding: above, below, left, right
    # Do all points at once using np.roll method
    next_grid = 0.25*(np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                      np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    
    load_boundary(next_grid)
        
    diff = abs(next_grid - grid)
    if np.all(diff <= 0.0001) == True:
        break
    np.copyto(grid, next_grid)
    curr_iter += 1
print("done")
dt = time.perf_counter() - t0
print("time taken:", dt, "sec")
    
# Now have numerical solution to Laplaces equation stored in next_grid
# Plot solution as 3D surface plot:
### Each point in the grid is a point in the x,y plane
### Each entry in the grid is the Voltgae at that point
### Plotting V(x,y) vs. (x,y)
xvals = np.linspace(XMIN, XMAX, NPTS)
yvals = np.linspace(YMIN, YMAX, NPTS)
X, Y = np.meshgrid(xvals, yvals)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, next_grid)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('V(x,y)')
plt.title('Approximate Potential')
plt.show()

