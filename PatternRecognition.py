import numpy as np
import scipy

from loadMNIST import loadMINST

from pylab import *

from random import randint
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom



with open('../testdata/mnist/t10k-images-idx3-ubyte', 'rb') as f:
    mn = loadMINST(f)
    n = 10000
    firstn = mn[:n,:,:]
    


#hold(True)
#imshow(firstn[0], origin='lower', extent=[1,28,1,28])

#show()

#print(firstn[0])


class Patterns:
    WHITE = -1000
    BLACK = 100
    IRREL = 0
    POS_WHITE = -1
    
    pe = np.array([[IRREL, WHITE, WHITE], [BLACK, IRREL, WHITE], [IRREL, BLACK, IRREL]])
    pf = np.array([[IRREL, BLACK, BLACK], [WHITE, IRREL, BLACK], [WHITE, WHITE, IRREL]])
    pg = np.array([[WHITE, BLACK, WHITE], [WHITE, IRREL, BLACK], [WHITE, WHITE, WHITE]])
    ph = np.array([[IRREL, BLACK, IRREL], [BLACK, IRREL, WHITE], [IRREL, WHITE, WHITE]])
    pi = np.array([[WHITE, WHITE, IRREL], [WHITE, IRREL, BLACK], [IRREL, BLACK, BLACK]])
    pj = np.array([[WHITE, WHITE, WHITE], [WHITE, IRREL, BLACK], [WHITE, BLACK, WHITE]])
    
    pk = np.array([[WHITE, WHITE, WHITE], [WHITE, IRREL, WHITE], [BLACK, BLACK, BLACK]])
    pl = np.array([[BLACK, WHITE, WHITE], [BLACK, IRREL, WHITE], [BLACK, WHITE, WHITE]])
    pm = np.array([[BLACK, BLACK, BLACK], [WHITE, IRREL, WHITE], [WHITE, WHITE, WHITE]])
    pn = np.array([[WHITE, WHITE, BLACK], [WHITE, IRREL, BLACK], [WHITE, WHITE, BLACK]])

    pa = np.array([[BLACK, BLACK, POS_WHITE],
                   [BLACK, IRREL, WHITE],
                   [BLACK, BLACK, POS_WHITE]])
    
    pb = np.array([[BLACK, BLACK, BLACK],
                  [BLACK, IRREL, BLACK],
                  [POS_WHITE, WHITE, POS_WHITE]])
    
    pc = np.array([[POS_WHITE, BLACK, BLACK, IRREL],
                   [WHITE, IRREL, BLACK, BLACK],
                   [POS_WHITE, BLACK, BLACK, IRREL]])
    
    pd = np.array([[POS_WHITE, WHITE, POS_WHITE],
                   [BLACK, IRREL, BLACK],
                   [BLACK, BLACK, BLACK],
                   [IRREL, BLACK, IRREL]])

    threshold = [(pe, 200), (pf, 300), (pg, 200), (ph, 200), (pi, 300), (pj, 200), (pk, 300), (pl, 300),
                 (pm, 300), (pn, 300), (pa, 499), (pb, 499)]
    
    threshold_expand = dict({'pc':599, 'pd':599})
    
    def find(self, target):
        if len(target) == 4:
            #check pattern d
            sum = 0
            for i in range(len(target)):
                sum += np.correlate(target[i], self.pd[i])[0]
            if sum >= self.threshold_expand['pd']:
                return True
            else:
                return False
            
        elif len(target) == 3:
            if len(target[0]) == 4:
                #check pattern c
                sum = 0
                for i in range(len(target)):
                    sum += np.correlate(target[i], self.pc[i])[0]
                if sum >= self.threshold_expand['pc']:
                    return True
                else:
                    return False
            elif len(target[0]) == 3:
                # common cases
                for k in self.threshold:
                    sum = 0
                    pt = k[0]
                    tr = k[1]
                    for i in range(len(target)):
                        sum += np.correlate(target[i], pt[i])[0]
                    if sum >= tr:
                        return True
                return False


def thinner(image):
    height = len(image)
    width = len(image[0])
    pattern_matcher = Patterns()
    output = image
    flag = True
    
    while (flag):
        thin = output
        flag = False
        for h in range(height-1):
            for w in range(width-1):
                if h!=0 and w!=0 and w!=width-2 and h!=height-2:
                    if output[h][w] == 1:
                        mat1 = np.array([[output[h-1][w-1], output[h-1][w], output[h-1][w+1]],
                                         [output[h][w-1], output[h][w], output[h][w+1]],
                                         [output[h+1][w-1], output[h+1][w], output[h+1][w+1]]])
                        if pattern_matcher.find(mat1):
                            thin[h][w] = 0
                            flag = True
                        else:
                            mat2 = np.array([[output[h-1][w-1], output[h-1][w], output[h-1][w+1], output[h-1][w+2]],
                                             [output[h][w-1], output[h][w], output[h][w+1], output[h][w+2]],
                                             [output[h+1][w-1], output[h+1][w], output[h+1][w+1], output[h+1][w+2]]])
                            if pattern_matcher.find(mat2):
                                thin[h][w] = 0
                                flag = True
                            else:
                                mat3 = np.array([[output[h-1][w-1], output[h-1][w], output[h-1][w+1]],
                                                 [output[h][w-1], output[h][w], output[h][w+1]],
                                                 [output[h+1][w-1], output[h+1][w], output[h+1][w+1]],
                                                 [output[h+2][w-1],output[h+2][w],output[h+2][w+1]]])
                                if pattern_matcher.find(mat3):
                                    thin[h][w] = 0
                                    flag = True
        output = thin
    return output


#print(firstn)
k=randint(0,10000)
test = zoom(np.array(firstn[k], dtype='double'), 3)
zeros = test < 255 * 0.6
test[ zeros] = 0
test[-zeros] = 1

#imshow(test, origin='lower', extent=[1,28,1,28])


#show()



result = thinner(np.copy(test))
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.imshow(result, interpolation='nearest')
ax2.imshow(test, interpolation='nearest')


show()
#imshow(test, origin='lower', extent=[1,28,1,28])


#imshow(result, origin='lower', extent=[1,28,1,28])

#show()



#print(result)


                            
            















