# cegaSim.py
# script to simulate molecular motors
# Author: PKR, July 2020

import numpy as np
from tkinter import filedialog
from skimage import io
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import os
import sys
import scipy.special as sp

# Choose the file for a background
FileLoc = filedialog.askopenfilename(initialdir = os.getcwd(),
            title = "Select the molecular motor movie",
            filetypes = (("tif","*.tif"),("tiff","*.tiff")))

# Set random number generator seed
prng = np.random.RandomState(seed=8675309)

# Experimental Parameters
gain = 70 # effective photon gain (ADU/e-)
offset = 2293 # ADU value added to image
WinLength = 31 # temporal window for median filter, must be an odd number
# Simulation Parameters
N = [5,5] # number of motors for each population on the image at any given time
D = [0.5,0.5] # diffusion rate by population, start with isotropic for now
V = [0.63,0.59] # mean speed of each motor population
VMax = [2.0, 2.0] # maximum speed of motor populations (confined process)
InitSigV = [0.5, 0.5] # lognormal sigma of the mean speed for populations
TimeSigV = [0.01, 0.01] # log normal sigma of the change in population speeds
# Photon Painting Parameters, Uniform across populations
PSFsigma = 0.75 # psf sigma in pixels, slightly undersampeld
MeanPhotons = 150 # photons per motor per frame
DBridge = 0.5 # diffusion constant of the brownian bridge between observations
WhiteNoiseSigma = 0.1 # sigma for white noise to make everything a little more realistic

# get number of populations from the N parameter
numPops = len(N)

# Load tiff files into numpy data structures
IM1 = io.imread(FileLoc)
# Gain and offset adjustment
im = (IM1-offset)/gain
im[im <= 0] = 0.01 # No negative values
sz = im.shape
# create a background movie with the temporal filter
HalfWin = WinLength//2
bg = 0*im
for ii in range(HalfWin):
    bg[ii] = np.median(im[:ii+HalfWin],axis=0)
for ii in range(HalfWin,sz[0]-HalfWin):
    bg[ii] = np.median(im[ii-HalfWin:ii+HalfWin],axis=0)
for ii in range(sz[0]-HalfWin,sz[0]):
    bg[ii] = np.median(im[ii-HalfWin:],axis=0)

# generate a set of tracks for the moving motors
# when a motor leaves the ROI, kill the track and start a new one
numFrames = sz[0]
yMax = sz[1] # get ROI boundaries
xMax = sz[2]
# pick the starting positions and velocities for motors
NStarts = []
vNStarts = []
for ii in range(numPops):
    NStarts.append(prng.random((N[ii],2))*[yMax,xMax])
    vNStarts.append(prng.lognormal(np.log(V[ii]),InitSigV[ii],(N[ii],2)))
    vNStarts[ii][NStarts[ii]>[yMax//2,xMax//2]] *= -1
# track trajectories, measured before and after each frame
NTracks = []
Nactive = []
# Start initial track structures
for ii in range(numPops):
    tempTracks = []
    for jj in range(N[ii]):
        tempTracks.append(np.concatenate([NStarts[ii][jj],[0],vNStarts[ii][jj]]))
        tempTracks[-1] = tempTracks[-1][np.newaxis,:]
    NTracks.append(tempTracks)
    Nactive.append([set() for tt in range(sz[0]+1)])
    Nactive[ii][0] = set(np.arange(N[ii]))

# loop over the entire movie length and generate tracks
nN = [] # keep track of the next track element
for ii in range(numPops):
    nN.append(N[ii])

for ii in range(numPops):
    for tt in range(1,sz[0]+1):
        for elem in Nactive[ii][tt-1]:
            nCoordinates = np.copy(NTracks[ii][elem][-1])
            # drift /w isotropic diffusion
            nCoordinates[0] += nCoordinates[3] + np.sqrt(2*D[ii])*prng.randn()
            nCoordinates[1] += nCoordinates[4] + np.sqrt(2*D[ii])*prng.randn()
            nCoordinates[2] = tt
            # isotropic confined diffusive process for velocity
            nCoordinates[3] += TimeSigV[ii]*prng.randn()
            nCoordinates[3] = max(-VMax[ii],min(VMax[ii],nCoordinates[3]))
            nCoordinates[4] += TimeSigV[ii]*prng.randn()
            nCoordinates[4] = max(-VMax[ii],min(VMax[ii],nCoordinates[4]))
            NTracks[ii][elem] = np.append(NTracks[ii][elem],nCoordinates[np.newaxis,:],axis=0)
            # kill a track if it wanders out of ROI, start a new track
            if (nCoordinates[0] < 0 or nCoordinates[0] > sz[1]) or (
                nCoordinates[1] < 0 or nCoordinates[1] > sz[2]):
                Nactive[ii][tt].add(nN[ii])
                newNStart = prng.random(2)*[yMax,xMax]
                newNVelocity = prng.lognormal(np.log(V[ii]),InitSigV[ii],2)
                newNVelocity[newNStart>[yMax//2,xMax//2]] *= -1
                NTracks[ii].append(np.concatenate([newNStart,[tt],newNVelocity]))
                NTracks[ii][-1] = NTracks[ii][-1][np.newaxis,:]
                nN[ii]+=1
            else:
                Nactive[ii][tt].add(elem)

# get emission times of photons for each blob
emissionTimes = []
for ii in range(numPops):
    emissionPop = []
    for mm in range(nN[ii]):
        startTime = NTracks[ii][mm][0][2]
        endTime = NTracks[ii][mm][-1][2]
        TimeGap = endTime-startTime
        sampleSize = int(MeanPhotons*TimeGap+5*np.sqrt(MeanPhotons*TimeGap))
        timeCDFs = prng.random(sampleSize)
        # convert CDFs to emission times
        xx = -np.log(1-timeCDFs)/MeanPhotons
        xx = np.cumsum(xx)+startTime
        xx = xx[xx < endTime] # remove emissions that go beyond the lifetime of the particle
        emissionPop.append(xx)
    emissionTimes.append(emissionPop)

# paint photons from tracks and emission times vectors, output a new tracks array and photon movie
motorPhotons = np.zeros(sz)
SimTracks = []
SimLocalizations = []

for ii in range(numPops):
    for mm in range(nN[ii]):
        cc = 0 # counter for emission times
        trackStartTime = int(NTracks[ii][mm][0][2])
        numEmissions = len(emissionTimes[ii][mm])
        PhotonDeviations = PSFsigma*prng.randn(numEmissions,2)
        BBridgeDraws = prng.randn(numEmissions,2)
        SimTracks.append([])
        while cc < numEmissions:
            startFrameEmission = cc
            tt = int(emissionTimes[ii][mm][cc])
            flag = True
            while flag: 
                cc+=1
                try:
                    flag = int(emissionTimes[ii][mm][cc]) == tt
                except:
                    flag = False
            startPos = NTracks[ii][mm][tt-trackStartTime][0:2]
            endPos = NTracks[ii][mm][tt+1-trackStartTime][0:2]
            # get time gaps between emission events
            emitFrame = emissionTimes[ii][mm][startFrameEmission:cc]-tt
            dtau = emitFrame-np.concatenate([[0], emitFrame[:-1]])
            endtau = 1-emitFrame[-1]
            # create Brownian Bridge Deviations
            BBridgeDeviations = np.sqrt(dtau[:,np.newaxis]
                *2*DBridge)*BBridgeDraws[startFrameEmission:cc]
            particlePosition = np.cumsum(BBridgeDeviations,axis=0)+startPos
            # include a fictional sample jump at the end of the frame
            maxDisplacement = np.sqrt(endtau*2*DBridge)*prng.randn(1,2)+particlePosition[-1]-endPos
            # brownian bridge formalism locks the start and end positions
            particlePosition -= emitFrame[:,np.newaxis]*maxDisplacement
            # draw photon positions from PSFsigma deviation
            photonPosition = particlePosition + PhotonDeviations[startFrameEmission:cc]
            # throw out photons that are outside the ROI
            photonPosition = photonPosition[photonPosition[:,0]>0]
            photonPosition = photonPosition[photonPosition[:,1]>0]
            photonPosition = photonPosition[photonPosition[:,0]<sz[1]]
            photonPosition = photonPosition[photonPosition[:,1]<sz[2]]
            frameCoordinates = [np.mean(photonPosition[:,0]),
                np.mean(photonPosition[:,1]),tt,len(photonPosition)]
            # put photon averages into the simulated tracks and localization structures
            SimLocalizations.append(frameCoordinates)
            SimTracks[-1].append(frameCoordinates)
            # paint the motorPhotons movie
            pixelPosition = photonPosition.astype(int)
            for py,px in pixelPosition:
                motorPhotons[tt,py,px] += 1

# create poisson-like realization of the background
bgRealization = prng.poisson(bg)
finalImage = WhiteNoiseSigma*prng.randn(sz[0],sz[1],sz[2])+bgRealization+motorPhotons

# sum motor photons with a poisson-realization of the background movie
pg.image(finalImage)
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()

