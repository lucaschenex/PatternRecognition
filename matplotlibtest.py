from pylab import *
x, y = ogrid[-1:1:0.01,-1:1:0.01]
z = sin(x) * y
hold(True)
imshow(z, origin='lower', extent=[-1,1,-1,1])

show()

#savefig() #name



