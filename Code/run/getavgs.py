import numpy as np
import pickle
import sys


tst_size = 300
_size = int(sys.argv[2])
a = pickle.load(open(sys.argv[1], 'r'))
avgs =  np.diag(np.mean(a[_size], axis=0))/tst_size

print (avgs, np.mean(avgs))
