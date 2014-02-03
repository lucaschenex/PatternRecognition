from loadMNIST import loadMINST

with open('../testdata/mnist/t10k-images-idx3-ubyte', 'rb') as f:
    mn = loadMINST(f)
    n = 1
    firstn = mn[:n,:,:]

print(firstn[0])

