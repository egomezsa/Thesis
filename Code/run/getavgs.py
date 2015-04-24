import numpy as np
import pickle
import os
import sys


directory = os.listdir('.')

tst_size = 100
_size = int(sys.argv[1])

writef = open( str(_size) +'.csv', 'w')
best = 0
bestType = None
for f in directory:
    if 'py' in f or 'csv' in f or 'back' in f:
        continue
    else:
        print f
        a = pickle.load(open(f,'r'))
        avgs = np.diag(np.mean(a[_size],axis=0))/tst_size
        if best < np.mean(avgs):
            best = np.mean(avgs)
            bestType = f
            print (bestType, best)
        #print avgs, np.mean(avgs)
        writef.write(f[:-2] + ', ' +str(avgs[0]) +  ', ' + str(avgs[1]) + ', ' + str(avgs[2]) + '\n')
print (bestType, best)
writef.close()

