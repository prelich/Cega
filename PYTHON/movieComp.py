# movieComp.py
# script to compare two data sets as a pyqtgraph object side by side
# Author: PKR, June 2020

import numpy as np
from tkinter import filedialog
from skimage import io
import pyqtgraph as pg
import os

# Chose the two files
FirstFile = filedialog.askopenfilename(initialdir = os.getcwd(),
            title = "Select the first tiff movie", 
            filetypes = (("tif","*.tif"),("tiff","*.tiff")))
SecondFile = filedialog.askopenfilename(initialdir = os.getcwd(),
             title = "Select the second tiff movie", 
             filetypes = (("tif","*.tif"),("tiff","*.tiff")))

# Load tiff files into numpy data structures
IM1 = io.imread(FirstFile)
IM2 = io.imread(SecondFile)
IMZ = np.concatenate((IM1,IM2),axis=1)
#IMD = IM1-IM2
pg.image(IMZ)
