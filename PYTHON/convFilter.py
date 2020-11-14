# convFilter.py
# a script to generate a convolution filter and plot it

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# generate three matrices of Discrete Gaussians with sig=0,1,2

Zer = np.zeros((5,5))

Zer[2,2] = 1

def GKernel(sigma=1,sz=5):
   K0 = sp.erf((np.arange(sz)-0.5-sz//2)/sigma/np.sqrt(2))
   K1 = sp.erf((np.arange(sz)+0.5-sz//2)/sigma/np.sqrt(2))
   return (K1-K0)/2

One = np.outer(GKernel(),GKernel())
One /= np.sum(One)
Two = np.outer(GKernel(2),GKernel(2))
Two /= np.sum(Two)

Weights = sp.erf((np.arange(3)+0.5)/np.sqrt(2))-sp.erf((np.arange(3)-0.5)/np.sqrt(2))
Weights /= sum(Weights) # normalize to 1

Zer *= Weights[0]/2
One *= Weights[1]/2
Two *= Weights[2]/2

# Print 5 histograms on top of each other, adjust in vector program

x = np.array([[i] * 5 for i in range(5)]).ravel() # x coordinates of each bar
y = np.array([i for i in range(5)] * 5) # y coordinates of each bar
z = np.zeros(25) # z coordinates of each bar
dx = np.ones(25) # length along x-axis of each bar
dy = np.ones(25) # length along y-axis of each bar
dz = One.ravel() # length along z-axis of each bar (height)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.bar3d(x, y, z, dx, dy, dz)
