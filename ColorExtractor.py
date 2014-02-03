import numpy as np
import scipy
import matplotlib.image as mpimg
import scipy
import plotting as pl
#import ndarray

from pylab import *
from random import randint
from matplotlib import pyplot as plt
from scipy.linalg import inv

import Image
import colorsys
import sys

# hsv to rgb conversion:
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def hsv_to_rgb_convert(img):
    """
    convert hsv space img to rgb space
    """
    h, s, v, a = np.rollaxis(img, axis = -1)
    r, g, b = hsv_to_rgb(h, s, v)
    out_arr = np.dstack((r, g, b, a))
    return out_arr

def clr_split(img, key):
    """
    default elm = 4;
    split rgb space img to three splited images
    """
    assert(img.any()),"empty img input"
    assert key in ["red", "green", "blue", "alpha"], "invalid key"
    
    
    dim1 = len(img)
    dim2 = len(img[0])
    output = np.empty((dim1, dim2))
    
    if key == 'red':
        output[:,:] = img[:,:,0]
    elif key == 'green':
        output[:,:] = img[:,:,1]
    elif key == 'blue':
        output[:,:] = img[:,:,2]
    elif key == 'alpha':
        output[:,:] = img[:,:,3]
    else:
        assert False, "Invalid key."
#    
#    for i in range(dim1):
#        for j in range(dim2):
#            if key == "red":
#                output[i][j] = [img[i][j][0], 0, 0, 255]
#            elif key == "green":
#                output[i][j] = [0, img[i][j][1], 0, 255]
#            elif key == "blue":
#                output[i][j] = [0, 0, img[i][j][2], 255]
#            else:
#                print("Error, invalid key")
#                break
    return output
    
    
# YPbPr to rgb conversion:


def YPbPr2RGB_float(YPbPr):
    offset = np.array([16, 128, 128])
    excursion = np.array([219, 224, 224])
    Kr = 0.299
    Kb = 0.114
    M = array([
        [1, 0, 2*(1-Kr)],
        [1, -2*(1-Kb)*Kb/(1-Kb-Kr), -2*(1-Kr)*Kr/(1-Kb-Kr)],
        [1, 2*(1-Kb), 0]
        ])
    M_inv = inv(M)
    YPbPr = np.array(YPbPr)
    return np.dot(M, YPbPr)
    
def YPbPr_to_rgb_convert(img):
    Y, Pb, Pr, a = np.rollaxis(img, axis = -1)
    r, g, b = YPbPr2RGB_float([Y, Pb, Pr])
    out_arr = np.dstack((r, g, b))
    return out_arr
    
def get_images(img, space_key):
    alpha = clr_split(img, "alpha")
    bgcolor = np.outer(np.array([0.5,0.5,0.5]), np.abs(alpha-255)).reshape((3,img.shape[0],img.shape[1]))
    
    if space_key == "rgb":
        return [clr_split(img, "red") * alpha + bgcolor[0,:,:],
                clr_split(img, "green") * alpha + bgcolor[1,:,:],
                clr_split(img, "blue") * alpha + bgcolor[2,:,:]]
    if space_key == "hsv":
        img = hsv_to_rgb_convert(img)
        return [clr_split(img, "red"), clr_split(img, "green"), clr_split(img, "blue")]
    elif space_key == "ypbpr":
        img = YPbPr_to_rgb_convert(img)
        return [clr_split(img, "red"), clr_split(img, "green"), clr_split(img, "blue")]
    elif space_key == "help":
        print("legal space_keys are 'rgb', 'hsv', and 'ypbpr'")
    else:
        print("Error, call get_images('help') for uage")

    
img = mpimg.imread("rgb.png")
#main
if len(sys.argv) == 3:
    #file is default to use "rgb.png" for test
    file = sys.argv[2]
    img = mpimg.imread(file)
    if sys.argv[1] == "-rgb":
        out_imgs = get_images(img, "rgb") #[clr_split(img, "red"), clr_split(img, "green"), clr_split(img, "blue"), clr_split(img, "alpha")]
    elif sys.argv[1] == "-hsv":
        out_imgs = get_images(img, "hsv")
    elif sys.argv[1] == "-ypbpr":
        out_imgs = get_images(img, "ypbpr")
    else:
        print "Invalid option"
        print "Use -rgb, -hsv or -ypbpr for color space"
        sys.exit()
    pl.plot_all(out_imgs)
    plt.show()
else:
    print "Invalid number of arguments."
    print "Use -rgb, -hsv or -ypbpr for color space, identify file name after that"
    




        
