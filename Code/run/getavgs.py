import numpy as np
import pickle
import os
import sys


directory = os.listdir('.')

tst_size = 300
_size = int(sys.argv[1])

writef = open('out.csv', 'w')
for f in directory:
    if 'py' in f or 'csv' in f or 'back' in f:
        continue
    else:
        print f
        a = pickle.load(open(f,'r'))
        avgs = np.diag(np.mean(a[_size],axis=0))/tst_size
        print avgs, np.mean(avgs)
        writef.write(f[:-2] + ', ' +str(avgs[0]) +  ', ' + str(avgs[1]) + ', ' + str(avgs[2]) + 'Total: ' + str(np.mean(avgs)) + '\n')

writef.close()

