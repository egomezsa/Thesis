import numpy as np
import pickle
import sys

a = pickle.load(open(sys.argv[1], 'r'))
print np.diag(np.mean(a[500], axis=0))
