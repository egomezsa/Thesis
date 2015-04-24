import numpy as np
import pickle
import os
import sys


directory = os.listdir('.')

tst_size = 100
_name = sys.argv[1]
_size = int(sys.argv[2])

writef = open( _name + '.csv', 'w')
a = pickle.load(open(_name,'r'))
avgs = np.mean(a[_size],axis=0)


for r in range(avgs.shape[0]):
    writef.write(str(avgs[r,0]) + ', ' + str(avgs[r,1]) + ', ' + str(avgs[r,2]) + '\n')

writef.close()

