# Cega
Spot Finder for Live Cell Microscopy Data, leverages temporal information to identify diffraction limited particles in noisy data

The MATLAB code is the most recent version of Ce:ga as used in the manuscript.
The PYTHON code is provided for portability purposes, but is not current with the changes in the MATLAB code so we are not liable for deviations between the two versions.

## To use the MATLAB version,
put cega.m in your path. <br>
Then call the function as [coordinates, filterMoivies] = cega( movie, ConnectivityThreshold, windowLength, sigmas, KLThreshold) <br>
Where: 
       - Movie is your gain corrected movie data
       - ConnectivityThreshold is usually set to around 0.1 or lower, its a threshold for adjacent pixels to denoise the filter 
       - windowLength is the length of the temporal median filter, we used 31 for our movies
       - sigmas are the standard deviations of the LoG filter, we used 1 and 1.5, but you can use any arbitrary number of scales
       - KLThreshold is the threshold of the KL divergence of the max pixel in the neighborhood
Try out testCega.m, it will plot the results of cega.m in matlab for you.

## Other MATLAB files of interest:
findSpots.m, the spot finder we used to compare with Cega, basically the LoG filter without the KL divergence step.

cegaSim.m, convert your real data into a simulation with ground truth values.  Just run the file, it will do the rest.

motorSim.m, cegaSim as a class object so you can imbed it into fancier scripts (used in scoreCega.m).

scoreCega.m, script to get the Jaccard Index between Cega and the Spot Finder
       - lap.m, mexlap.m and all the .mex files are needed by scoreCega.m (linear assignment problem code)
## To use the PYTHON version
cega.py , run it and it will make the filter movies as well as the coordinates in a dictionary.  The python version does not have a final thresholding step (The KLThresh function) on the spot candidates.

motorSim.py, same as motorSim.m, a class object for generating molecular motor simulations
