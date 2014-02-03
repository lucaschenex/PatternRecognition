import numpy as np
import scipy
import matplotlib.image as mpimg
import scipy.ndimage

#from loadMNIST import loadMINST
from pylab import *
from random import randint
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom


#im = Image.open("bug.png")
#imv = im.load()

im = Image.open("Programmer.jpg")
imv = im.load()
img=mpimg.imread('bug.png')



#r,g,b = im.split()
#
#r.show(r)
#g.show(g)
#b.show(b)
#im1=Image.merge("RGB", (r, g, b))
#im1.show(im1)

