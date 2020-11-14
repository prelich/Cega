# motorSim.py
# class module for molecular motors

import numpy as np
from tkinter import filedialog
from skimage import io
import os
import sys
import scipy.special as sp

class motorSim:
    def __init__(self):
        # output data structures
        self.bg = []
        self.im = []
        self.sz = []
        self.SimTracks = []
        self.SimLocalizations = []
        self.SimMovie = []
        # intermediate data structures
        self.nN = []
        self.NTracks = []
        self.Nactive = []
        self.emissionTimes = []
        self.motorPhotons = []
        # experimental parameters
        self.gain = 70 # effective photon gain (ADU/e-)
        self.offset = 2293 # ADU value added to image
        self.WinLength = 7 # temporal window for median filter, must be odd
        # simulation parameters
        self.N = [10,10] # number of motors for each population on the image at any frame
        self.D = [0.5,0.5] # diffusion rate by population, isotropic diffusion
        self.V = [0.63, 0.59] # mean speed of each motor population
        self.VMax = [2.0,2.0] # maximum speed of motor populations (confined max)
        self.InitSigV = [0.5,0.5] # lognormal sigma of population mean speed
        self.TimeSigV = [0.01,0.01] # normal sigma of the change in population speedi
        # Photon Painting Parameters, Uniform across populations
        self.PSFsigma = 0.75 # psf sigma in pixels
        self.MeanPhotons = 250 # photons per motor per frame
        self.DBridge = 0.5 # diffusion constant of the brownian bridge interpolator
        self.WhiteNoiseSigma = 0.1 # sigma for white noise in image
        # number of populations from N length
        self.numPops = len(self.N)
        # set the default seed and random number generator
        self.seed = 8675309
        self.setRNG()

    def buildDataSet(self):
        self.prepBackground()
        self.getRoughTracks()
        self.getEmissionTimes()
        self.paintSim()

    def loadExperiment(self):
        # Choose the file to load
        FileLoc = filedialog.askopenfilename(initialdir = os.getcwd(),
            title = "Select the molecular motor movie",
            filetypes = (("tif","*.tif"),("tiff","*.tiff")))
        IM1 = io.imread(FileLoc)
        self.im = (IM1-self.offset)/self.gain # gain/offset adjustment
        self.im[self.im <= 0] = 0.01 # no negative values
        self.sz = self.im.shape

    def prepBackground(self):
        if not len(self.sz):
            self.loadExperiment()
        # create a background muovie with the temporal filter
        HalfWin = self.WinLength//2
        self.bg = 0*self.im
        for ii in range(HalfWin):
            self.bg[ii] = np.median(self.im[:ii+HalfWin],axis=0)
        for ii in range(HalfWin,self.sz[0]-HalfWin):
            self.bg[ii] = np.median(self.im[ii-HalfWin:ii+HalfWin],axis=0)
        for ii in range(self.sz[0]-HalfWin,self.sz[0]):
            self.bg[ii] = np.median(self.im[ii-HalfWin:],axis=0)

    def getRoughTracks(self):
        if not len(self.bg):
            self.prepBackground()
        # zero out relevant intermediate variables
        self.NTrakcs = []
        self.Nactive = []
        # generate a set of tracks for the moving motors
        # get ROI boundaries and frame times
        numFrames = self.sz[0]
        yMax = self.sz[1]
        xMax = self.sz[2]
        # pick starting positions and velocities for motors
        NStarts = []
        vNStarts = []
        for ii in range(self.numPops):
            NStarts.append(self.prng.random((self.N[ii],2))*[yMax,xMax])
            vNStarts.append(self.prng.lognormal(np.log(self.V[ii]),
                self.InitSigV[ii],(self.N[ii],2)))
            vNStarts[ii][NStarts[ii]>[yMax//2,xMax//2]] *= -1
        # track trajectories, measured between exposure times
        # start initial track structures
        for ii in range(self.numPops):
            tempTracks = []
            for jj in range(self.N[ii]):
                tempTracks.append(np.concatenate([NStarts[ii][jj],[0],vNStarts[ii][jj]]))
                tempTracks[-1] = tempTracks[-1][np.newaxis,:]
            self.NTracks.append(tempTracks)
            self.Nactive.append([set() for tt in range(self.sz[0]+1)])
            self.Nactive[ii][0] = set(np.arange(self.N[ii]))
        # loop over the entire movie length and generate tracks
        self.nN = [] # keep track of the next track element
        for ii in range(self.numPops):
            self.nN.append(self.N[ii])
        # build tracks for each pop frame by frame
        for ii in range(self.numPops):
            for tt in range(1,self.sz[0]+1):
                for elem in self.Nactive[ii][tt-1]:
                    nCoordinates = np.copy(self.NTracks[ii][elem][-1])
                    # drift /w isotropic diffusion
                    nCoordinates[0] += nCoordinates[3] + np.sqrt(2*self.D[ii])*self.prng.randn()
                    nCoordinates[1] += nCoordinates[4] + np.sqrt(2*self.D[ii])*self.prng.randn()
                    nCoordinates[2] = tt
                    # isotropic confined diffusive process for velocity
                    nCoordinates[3] += self.TimeSigV[ii]*self.prng.randn()
                    nCoordinates[3] = max(-self.VMax[ii],min(self.VMax[ii],nCoordinates[3]))
                    nCoordinates[4] += self.TimeSigV[ii]*self.prng.randn()
                    nCoordinates[4] = max(-self.VMax[ii],min(self.VMax[ii],nCoordinates[4]))
                    self.NTracks[ii][elem] = np.append(self.NTracks[ii][elem],
                        nCoordinates[np.newaxis,:],axis=0)
                    # kill a track if it wanders out of ROI, start a new track
                    if (nCoordinates[0] < 0 or nCoordinates[0] > self.sz[1]) or (
                        nCoordinates[1] < 0 or nCoordinates[1] > self.sz[2]):
                        self.Nactive[ii][tt].add(self.nN[ii])
                        newNStart = self.prng.random(2)*[yMax,xMax]
                        newNVelocity = self.prng.lognormal(np.log(self.V[ii]),self.InitSigV[ii],2)
                        newNVelocity[newNStart>[yMax//2,xMax//2]] *= -1
                        self.NTracks[ii].append(np.concatenate([newNStart,[tt],newNVelocity]))
                        self.NTracks[ii][-1] = self.NTracks[ii][-1][np.newaxis,:]
                        self.nN[ii]+=1
                    else:
                        self.Nactive[ii][tt].add(elem)

    def getEmissionTimes(self):
        # get emission times of photons for each blob
        self.emissionTimes = []
        for ii in range(self.numPops):
            emissionPop = []
            for mm in range(self.nN[ii]):
                startTime = self.NTracks[ii][mm][0][2]
                endTime = self.NTracks[ii][mm][-1][2]
                TimeGap = endTime-startTime
                sampleSize = int(self.MeanPhotons*TimeGap+5*np.sqrt(self.MeanPhotons*TimeGap))
                timeCDFs = self.prng.random(sampleSize)
                # convert CDFs to emission times
                xx = -np.log(1-timeCDFs)/self.MeanPhotons
                xx = np.cumsum(xx)+startTime
                xx = xx[xx < endTime] # remove emissions that go beyond the lifetime of the particle
                emissionPop.append(xx)
            self.emissionTimes.append(emissionPop)

    def paintSim(self):
        # initialize outputs
        self.motorPhotons = np.zeros(self.sz)
        self.SimTracks = []
        self.SimLocalizations = []
        for ii in range(self.numPops):
            for mm in range(self.nN[ii]):
                cc = 0 # emission time counter
                trackStartTime = int(self.NTracks[ii][mm][0][2])
                numEmissions = len(self.emissionTimes[ii][mm])
                PhotonDeviations = self.PSFsigma*self.prng.randn(numEmissions,2)
                BBridgeDraws = self.prng.randn(numEmissions,2)
                self.SimTracks.append([])
                while cc < numEmissions:
                    startFrameEmission = cc
                    tt = int(self.emissionTimes[ii][mm][cc])
                    flag = True
                    while flag:
                        cc+=1
                        try:
                            flag = int(self.emissionTimes[ii][mm][cc]) == tt
                        except:
                            flag = False
                    startPos = self.NTracks[ii][mm][tt-trackStartTime][0:2]
                    endPos = self.NTracks[ii][mm][tt+1-trackStartTime][0:2]
                    # get time gaps between emission events
                    emitFrame = self.emissionTimes[ii][mm][startFrameEmission:cc]-tt
                    dtau = emitFrame-np.concatenate([[0], emitFrame[:-1]])
                    endtau = 1-emitFrame[-1]
                    # create Brownian Bridge Deviations
                    BBridgeDeviations = np.sqrt(dtau[:,np.newaxis]
                        *2*self.DBridge)*BBridgeDraws[startFrameEmission:cc]
                    particlePosition = np.cumsum(BBridgeDeviations,axis=0)+startPos
                    # include a fictional sample jump at the end of the frame
                    maxDisplacement = (np.sqrt(endtau*2*self.DBridge)*self.prng.randn(1,2)
                        +particlePosition[-1]-endPos)
                    # brownian bridge formalism locks the start and end positions
                    particlePosition -= emitFrame[:,np.newaxis]*maxDisplacement
                    # draw photon positions from PSFsigma deviation
                    photonPosition = particlePosition + PhotonDeviations[startFrameEmission:cc]
                    # throw out photons that are outside the ROI
                    photonPosition = photonPosition[photonPosition[:,0]>0]
                    photonPosition = photonPosition[photonPosition[:,1]>0]
                    photonPosition = photonPosition[photonPosition[:,0]<self.sz[1]]
                    photonPosition = photonPosition[photonPosition[:,1]<self.sz[2]]
                    frameCoordinates = [np.mean(photonPosition[:,0]),
                        np.mean(photonPosition[:,1]),tt,len(photonPosition)]
                    # put photon averages into the simulated tracks and localization structures
                    self.SimLocalizations.append(frameCoordinates)
                    self.SimTracks[-1].append(frameCoordinates)
                    # paint the motorPhotons movie
                    pixelPosition = photonPosition.astype(int)
                    for py,px in pixelPosition:
                        self.motorPhotons[tt,py,px] += 1
        # create poisson-like realization of the background
        bgRealization = self.prng.poisson(self.bg)
        self.SimMovie = (self.WhiteNoiseSigma*self.prng.randn(self.sz[0],self.sz[1],self.sz[2])
            +bgRealization+self.motorPhotons)

    def setRNG(self):
        self.prng = np.random.RandomState(seed=self.seed)
