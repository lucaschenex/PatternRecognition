import numpy as np
import scipy
import matplotlib.image as mpimg
import scipy.ndimage

#from loadMNIST import loadMINST
from pylab import *
from random import randint
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom


#with open('../testdata/mnist/t10k-images-idx3-ubyte', 'rb') as f:
#    mn = loadMINST(f)
#    n = 10000
#    firstn = mn[:n,:,:]

img=mpimg.imread('bug.png')
print "Image dtype: %s" %(img.dtype)
print ''Image size: %6d''%(img.size)
print ''Image shape: %3dx%3d''%(img.shape[0],img.shape[1])
print ''Max value %1.2f at pixel %6d''%(img.max(),img.argmax())
print ''Min value %1.2f at pixel %6d''%(img.min(),img.argmin())
print ''Variance: %1.5f\nStandard deviation: %1.5f''%(img.var(),img.std())



def deletable(pointy, pointx, iter_n, image):
    """
    True or false
        - image is a np array of the whole graph
    """
    nb = {"p2": image[pointy-1][pointx], "p3": image[pointy-1][pointx+1],
          "p4": image[pointy][pointx+1], "p5": image[pointy+1][pointx+1],
          "p6": image[pointy+1][pointx], "p7": image[pointy+1][pointx-1],
          "p8": image[pointy][pointx-1], "p9": image[pointy-1][pointx-1]}
    nblist = ["p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p2"]
    
    def cond_a():
        BP1 = 0
        for p in nb:
            if nb[p] == 1:
               BP1 += 1
        if BP1 >=2 and BP1 <= 6:
            return True
        return False
    
    def cond_b():
        AP1 = 0
        flag = False
        for p in nblist:
            if flag:
                if nb[p] == 1:
                    AP1 += 1
                    flag = False
            else:
                if nb[p] == 0:
                    flag = True
        if AP1 == 1:
            return True
        return False

    def cond_c_d_1():
        condc = nb["p2"] * nb["p4"] * nb["p6"]
        condd = nb["p4"] * nb["p6"] * nb["p8"]
        if not condc and not condd:
            return True
        return False

    def cond_c_d_2():
        condc = nb["p2"] * nb["p4"] * nb["p8"]
        condd = nb["p2"] * nb["p6"] * nb["p8"]
        if not condc and not condd:
            return True
        return False
    
    if image[pointy][pointx] == 0:
        return False
    
    if iter_n == 1:
        if cond_a() and cond_b() and cond_c_d_1():
            return True
        return False
    elif iter_n == 2:
        if cond_a() and cond_b() and cond_c_d_2():
            return True
        return False
    else:
        print ("wrong iteration number")
        
def thinner(image):
    counter = 0
    output = image
    while True:
        output, counter = thinner_iter1(output, counter)
        if not counter:
            break
        counter = 0
        output, counter = thinner_iter2(output, counter)
        if not counter:
            break
    return output

def thinner_iter1(image, counter):
    output = image.copy()
    for py in range(1,len(image)-1):
        for px in range(1,len(image[0])-1):
            if deletable(py, px, 1, image):
                output[py][px] = 0
                counter += 1
    return output, counter
     
def thinner_iter2(image, counter):
    output = image.copy()
    for py in range(1,len(image)-1):
        for px in range(1,len(image[0])-1):
            if deletable(py, px, 2, image):
                output[py][px] = 0
                counter += 1
    return output, counter

#k=randint(0,10000)
#test = zoom(np.array(firstn[k], dtype='double'), 3)



test = zoom(np.array(img, dtype = 'float32'), 1)
print "after zoom"
print test
#threshold = (test != 0)
#print threshold
#test[threshold] = 1

#zeros = test < 255 * 0.6
#test[ zeros] = 0
#test[-zeros] = 1
#
#result = thinner(np.copy(test))
#result = thinner(test.all())
#fig = plt.figure()
#ax1 = fig.add_subplot(1,2,1)
#ax2 = fig.add_subplot(1,2,2)
#ax1.imshow(result, interpolation='nearest')
#ax2.imshow(test, interpolation='nearest')


#show()














    
        

