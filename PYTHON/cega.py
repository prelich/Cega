# cega: a function to take a motor movie and find the moving components
# Author: PKR UPENN June 2020
# ce:ga (Tohon O'odham: to find object)
# License: GPL-3.0-or-later

import numpy as np
import scipy.special as sp

# define filter functions here
# Kernel in use, sampled Gaussian
def GaussKernel(sig=1,sz=7):
    Kern = np.exp(-((np.arange(sz)-sz//2)**2)/2/sig/sig)
    return Kern/sum(Kern)
# error function kernel, good for blurring, but not ideal for scale space
def ErrKernel(sig=1,sz=7):
   K0 = sp.erf((np.arange(sz)-0.5-sz//2)/sig/np.sqrt(2))
   K1 = sp.erf((np.arange(sz)+0.5-sz//2)/sig/np.sqrt(2))
   return (K1-K0)/np.sum(K1-K0)
# tighter kernel than gauss, but I didn't have the LoG function for this one
def disGKernel(sig=1,sz=7):
    t = sig*sig
    Kern = np.exp(-t)*sp.iv(np.arange(sz)-sz//2,t)
    return Kern/sum(Kern)
# convolution function for separable filters
def filterKern2D(movie,kernelX,kernelY):
    kX = kernelX.shape[0]
    midX = kX//2
    kY = kernelY.shape[0]
    midY = kY//2
    filter1 = 0*movie
    filterOut = 0*movie
    mz = movie.shape
    # convolve along last dimension first
    for jj in range(kX):
        lb = max(midX-jj,0)
        rb = min(mz[-1],mz[-1]-jj+midX)
        ml = max(jj-midX,0)
        mr = min(mz[-1],mz[-1]+jj-midX)
        filter1[:,:,lb:rb] += movie[:,:,ml:mr]*kernelX[jj]
   # reflect edges
    for jj in range(midX-1):
        filter1[:,:,jj] += np.sum(
            filter1[:,:,:midX-jj]*kernelX[midX-jj-1::-1],axis=-1)
        filter1[:,:,-jj-1] += np.sum(
            filter1[:,:,-midX+jj:]*kernelX[-1:midX+jj:-1],axis=-1)
    # convolve along next dimension
    for jj in range(kY):
        lb = max(midY-jj,0)
        rb = min(mz[-2],mz[-2]-jj+midY)
        ml = max(jj-midY,0)
        mr = min(mz[-2],mz[-2]+jj-midY)
        filterOut[:,lb:rb,:] += filter1[:,ml:mr,:]*kernelY[jj]
    # reflect edges
    for jj in range(midY-1):
        filterOut[:,jj,:] += np.sum(
            filterOut[:,:midY-jj,:]*kernelY[midY-jj-1::-1,np.newaxis],axis=-2)
        filterOut[:,-jj-1,:] += np.sum(
            filterOut[:,-midY+jj:,:]*kernelY[-1:midY+jj:-1,np.newaxis],axis=-2)
    return filterOut

# connectivity filter for a 3D ndArray movie (0 axis is time)
def connectivityFilter(im,ConnThresh):
    ConnFilt = np.ndarray.copy(im) # make sure to deep copy this
    sz = ConnFilt.shape
    ThreshMovie = ConnFilt > ConnThresh
    for jj in range(sz[1]):
        # set left/right bounds for image border
        lb = jj-1 if jj > 0 else 0
        rb = jj+2 if jj < sz[1]-1 else sz[1]
        for kk in range(sz[2]):
            # set top/bottom bounds for image border
            tb = kk-1 if kk > 0 else 0
            bb = kk+2 if kk < sz[2]-1 else sz[2]
            # perform connectivity filter here
            tempValids = np.squeeze(ThreshMovie[:,jj,kk])
            ThreshChunk = ThreshMovie[tempValids,lb:rb,tb:bb]
            conn = np.sum(ThreshChunk,axis=(1,2)) 
            tempValids[tempValids] = conn > 3
            ConnFilt[np.invert(tempValids),jj,kk] = 0
    return ConnFilt

# additional functions for performing the LoG filter
def d2GaussKernel(sig=1,sz=7):
    gK = GaussKernel(sig,sz)
    # multiplied by extra sig**2 to make it scale invariant
    return [gK * ((np.arange(sz)-sz//2)**2-sig**2)/sig**2, gK]
def LoGFilter(movie, gKernel, d2gKernel):
    dxx = filterKern2D(movie,d2gKernel,gKernel)
    dyy = filterKern2D(movie,gKernel,d2gKernel)
    return dxx+dyy # Trace of the Hessian
def scaleSpaceLoG(movie, sigmas):
    mz = movie.shape
    tempLoG = [] 
    for ii in range(len(sigmas)):
        kz = int(6*sigmas[ii]+1)
        kz+=(kz+1)%2 # kernel lens must be odd numbers
        [d2gKernel,gKernel] = d2GaussKernel(sigmas[ii],sz=kz)
        tempLoG.append(LoGFilter(movie,gKernel,d2gKernel))
    return np.concatenate([LoG[np.newaxis] for LoG in tempLoG])
# return local maxima prior to thresholding
def findLoGMinima(movie, sigmas, order=1):
    LoGs = scaleSpaceLoG(movie, sigmas)
    lz = LoGs.shape
    Candidates = LoGs<0 # only bright spots are considered
    # order is neighbor size
    for jj in range(order):
        # x-axis
        Candidates[:,:,:,:-1-jj] *= (LoGs[:,:,:,:-1-jj] 
            < LoGs[:,:,:,jj+1:])
        Candidates[:,:,:,jj+1:] *= (LoGs[:,:,:,jj+1:]
            < LoGs[:,:,:,:-1-jj])
        # y-axis
        Candidates[:,:,:-1-jj,:] *= (LoGs[:,:,:-1-jj,:]      
            < LoGs[:,:,jj+1:,:])
        Candidates[:,:,jj+1:,:] *= (LoGs[:,:,jj+1:,:]
            < LoGs[:,:,:-1-jj,:])
        # scale axis
        Candidates[:-1-jj,:,:,:] *= (LoGs[:-1-jj,:,:,:]      
            < LoGs[jj+1:,:,:,:])
        Candidates[jj+1:,:,:,:] *= (LoGs[jj+1:,:,:,:]
            < LoGs[:-1-jj,:,:,:])
    # Loop over remaining elements and check all "surviving" nearest neighbors
    [s,t,y,x] = np.nonzero(Candidates)
    bMask = np.full(len(s),True)
    for ii in range(len(s)):
        val = LoGs[s[ii],t[ii],y[ii],x[ii]]
        s_l = max(0,s[ii]-order)
        s_r = min(lz[0],s[ii]+order)
        y_l = max(0,y[ii]-order)
        y_r = min(lz[-2],y[ii]+order)
        x_l = max(0,x[ii]-order)
        x_r = min(lz[-1],x[ii]+order)
        minVal = np.min(LoGs[s_l:s_r,t[ii],y_l:y_r,x_l:x_r])
        if minVal < val:
            bMask[ii] = False
    return [s[bMask],t[bMask],y[bMask],x[bMask]]

# Main function for finding maxima candidates
def cega(im, ConnThresh=0.5, WinLength=31, sigmas=[1,1.5]):
    # im should be gain calibrated or contrast will be off
    im[im <= 0] = 0.01 # no negative values
    sz = im.shape
    # create Gaussian filtered images, using GaussKernel here
    gKernel = GaussKernel()
    gKernel2 = GaussKernel(2,11)
    tKernel = GaussKernel(1,5)
    [d2Kernel,_] = d2GaussKernel()
    # Gaussian blurred movies for generating P and Q
    imG = filterKern2D(im,gKernel,gKernel)
    imG2 = filterKern2D(im,gKernel2,gKernel2)
    # generate <P>
    ppt = np.concatenate((np.arange(2,sz[0]), np.array([sz[0]-1, sz[0]-2])))
    pt = np.concatenate((np.arange(1,sz[0]), np.array([sz[0]-1])))
    mt = np.concatenate((np.array([0]),np.arange(sz[0]-1)))
    mmt = np.concatenate((np.array([1, 0]),np.arange(sz[0]-2)))
    P = (tKernel[2]*im+tKernel[1]*imG[mt]+tKernel[3]*imG[pt]
        +tKernel[0]*imG2[mmt]+tKernel[4]*imG2[ppt])
    # generate <Q>
    B_G = (tKernel[2]*im+tKernel[1]*imG+tKernel[3]*imG
        +tKernel[0]*imG2+tKernel[4]*imG2)
    Q = 0*im
    HalfWin = WinLength//2
    for ii in range(HalfWin):
        Q[ii] = np.median(B_G[:ii+HalfWin],axis=0)
    for ii in range(HalfWin,sz[0]-HalfWin):
        Q[ii] = np.median(B_G[ii-HalfWin:ii+HalfWin],axis=0)
    for ii in range(sz[0]-HalfWin,sz[0]):
        Q[ii] = np.median(B_G[ii-HalfWin:],axis=0)
    # generate KL Divergence Movie (KLM)
    KLM = P*(-np.log(Q)+np.log(P))-P+Q
    # run Connectivity filter
    ConnFilt = connectivityFilter(KLM,ConnThresh)
    # save out a LoG filter
    LoG = LoGFilter(ConnFilt,gKernel,d2Kernel)
    # output the maximas
    maximas = findLoGMinima(ConnFilt,sigmas)
    filterMovies = {'expectedQ':Q, 'expectedP':P, 'KLM':KLM, 
    'ConnectivityFilter':ConnFilt, 'LoG':LoG}
    return {'Maximas':maximas,'FilterMovies':filterMovies}

# test the script here
if __name__ ==  "__main__":
    from tkinter import filedialog
    from skimage import io
    import os
    import sys
    # need event loop for gui
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui

    FileLoc = filedialog.askopenfilename(initialdir = os.getcwd(),
          title = "Select tif movie to analyze",
          filetypes = (("tif","*.tif"),("tiff","*.tiff")))
    # load movie
    im = io.imread(FileLoc)
    # set CEGA parameters
    WinLength=31
    ConnThresh = 0.5
    sigmas = [1,1.5]
    # Gain/Offset Calibration
    Gain = 70
    Offset = 2293
    imM = (im-Offset)/Gain
    imM[imM<=0] = 0.01 # no 0 or negative values
    outDict = cega(imM,ConnThresh,WinLength,sigmas)
    FilterMovies = outDict['FilterMovies']
    # stack all filter movies on top of one another
    dispMovie = np.concatenate((imM, 
        FilterMovies['expectedP'],
        FilterMovies['expectedQ'],
        FilterMovies['KLM']*10,
        FilterMovies['ConnectivityFilter']*10,
        -FilterMovies['LoG']*10),axis=1)
    dispMovie = np.transpose(dispMovie,(0,2,1))
# output a movie
#    import skvideo.io
#    dispMovie+=20
#    dispMovie = dispMovie.astype(np.uint8)
#    skvideo.io.vwrite("cegavideo.mp4",dispMovie)
# pyqtgraph GUI output
    pg.image(dispMovie)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

