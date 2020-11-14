# cegaTest.py
# python test script to try different filter Kernels on Erin's data and look to see which result is best
# Author: PKR June 2020

import numpy as np
import scipy.special as sp
from tkinter import filedialog
from skimage import io
import pyqtgraph as pg
import os

# define filter functions here
def GaussKernel(sig=1,sz=7):
    Kern = np.exp(-((np.arange(sz)-sz//2)**2)/2/sig/sig)
    return Kern/sum(Kern)

def ErrKernel(sig=1,sz=7):
   K0 = sp.erf((np.arange(sz)-0.5-sz//2)/sig/np.sqrt(2))
   K1 = sp.erf((np.arange(sz)+0.5-sz//2)/sig/np.sqrt(2))
   return (K1-K0)/np.sum(K1-K0) 

def disGKernel(sig=1,sz=7):
    t = sig*sig
    Kern = np.exp(-t)*sp.iv(np.arange(sz)-sz//2,t)
    return Kern/sum(Kern)

def filterKern2D(movie,kernel):
    kz = kernel.shape[0]
    mid = kz//2
    filter1 = 0*movie
    filterOut = 0*movie
    mz = movie.shape
    # convolve along last dimension first
    for jj in range(kz):
        lb = max(mid-jj,0)
        rb = min(mz[-1],mz[-1]-jj+mid)
        ml = max(jj-mid,0)
        mr = min(mz[-1],mz[-1]+jj-mid)
        filter1[:,:,lb:rb] += movie[:,:,ml:mr]*kernel[jj]
    # normalize edges
    for jj in range(mid-1):
        norml = sum(kernel[mid-jj:])
        normr = sum(kernel[:-mid+jj])
        filter1[:,:,jj] = filter1[:,:,jj]/norml
        filter1[:,:,-jj-1] = filter1[:,:,-jj-1]/normr
    # convolve along next dimension
    for jj in range(kz):
        lb = max(mid-jj,0)
        rb = min(mz[-2],mz[-2]-jj+mid)
        ml = max(jj-mid,0)
        mr = min(mz[-2],mz[-2]+jj-mid)
        filterOut[:,lb:rb,:] += movie[:,ml:mr,:]*kernel[jj]
    # normalize edges
    for jj in range(mid-1):
        norml = sum(kernel[mid-jj:])
        normr = sum(kernel[:-mid+jj])
        filterOut[:,jj,:] = filterOut[:,jj,:]/norml
        filterOut[:,-jj-1,:] = filterOut[:,-jj-1,:]/normr
    return filterOut

# Choose the file
FileLoc = filedialog.askopenfilename(initialdir = os.getcwd(),
          title = "Select tif movie to analyze", 
          filetypes = (("tif","*.tif"),("tiff","*.tiff")))
# load the file
im = io.imread(FileLoc)
# Set parameters
WinLength = 31
ConnThresh = 1

# Gain adjust the movie
Gain = 70
Offset = 2293
imM = (im-Offset)/Gain
imM[imM<=0] = 0.01 # no negative values
sz = imM.shape

# create the various kernels
gsk = GaussKernel()
esk = ErrKernel()
dsk = disGKernel()

gsk2 = GaussKernel(2,11)
esk2 = ErrKernel(2,11)
dsk2 = disGKernel(2,11)

imG = filterKern2D(imM,gsk)
imE = filterKern2D(imM,esk)
imD = filterKern2D(imM,dsk)

imG2 = filterKern2D(imM,gsk2)
imE2 = filterKern2D(imM,esk2)
imD2 = filterKern2D(imM,dsk2)

# generate 3 <P>s
ppt = np.concatenate((np.arange(2,sz[0]), np.array([sz[0]-1, sz[0]-2])))
pt = np.concatenate((np.arange(1,sz[0]), np.array([sz[0]-1])))
mt = np.concatenate((np.array([0]),np.arange(sz[0]-1)))
mmt = np.concatenate((np.array([1, 0]),np.arange(sz[0]-2)))

P_G = 0.4026*imM+0.2442*imG[pt]+0.2442*imG[mt]+0.0545*imG2[ppt]+0.0545*imG2[mmt]
P_E = 0.3877*imM+0.2447*imE[pt]+0.2447*imE[mt]+0.0613*imE2[ppt]+0.0613*imE2[mmt]
P_D = 0.4745*imM+0.2118*imD[pt]+0.2118*imD[mt]+0.0509*imD2[ppt]+0.0509*imD2[mmt]

# generate 3 <Q>s
# first generate Gaussian blur profile similar to P movie
B_G = 0.4026*imM+2*0.2442*imG+2*0.0545*imG2
B_E = 0.3877*imM+2*0.2447*imE+2*0.0613*imE2
B_D = 0.4745*imM+2*0.2118*imD+2*0.0509*imD2
# Want to blur out moving stuff completely!
#B_G = 0.801*imG+0.199*imG2
#B_E = 0.801*imE+0.199*imE2
#B_D = 0.801*imD+0.199*imD2
#B_G = imG
#B_E = imE
#B_D = imD

# perform median filtering along time axis
Q_G = 0*imM
Q_E = 0*imM
Q_D = 0*imM

HalfWin = WinLength//2
for ii in range(HalfWin):
    Q_G[ii] = np.median(B_G[:ii+HalfWin],axis=0)
    Q_E[ii] = np.median(B_E[:ii+HalfWin],axis=0)
    Q_D[ii] = np.median(B_D[:ii+HalfWin],axis=0)
for ii in range(HalfWin,sz[0]-HalfWin):
    Q_G[ii] = np.median(B_G[ii-HalfWin:ii+HalfWin],axis=0)
    Q_E[ii] = np.median(B_E[ii-HalfWin:ii+HalfWin],axis=0)
    Q_D[ii] = np.median(B_D[ii-HalfWin:ii+HalfWin],axis=0)
for ii in range(sz[0]-HalfWin,sz[0]):
    Q_G[ii] = np.median(B_G[ii-HalfWin:],axis=0)
    Q_E[ii] = np.median(B_E[ii-HalfWin:],axis=0)
    Q_D[ii] = np.median(B_D[ii-HalfWin:],axis=0)

KL_G = P_G*(-np.log(Q_G)+np.log(P_G))-P_G+Q_G
KL_E = P_E*(-np.log(Q_E)+np.log(P_E))-P_E+Q_E
KL_D = P_D*(-np.log(Q_D)+np.log(P_D))-P_D+Q_D

pg.image(np.concatenate((imM,Q_G,P_G,10*KL_G,Q_E,P_E,10*KL_E,Q_D,P_D,10*KL_D),axis=1))
