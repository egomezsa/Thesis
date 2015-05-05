import numpy as np
import pickle
import os
import sys


directory = os.listdir('.')

tst_size = 100
_size = int(sys.argv[1])


writef = open( str(_size) +'.csv', 'w')

for f in directory:
    if 'py' in f or 'csv' in f or 'back' in f:
        continue
    else:
        print f
        a = pickle.load(open(f,'r'))

        avg_list = []

        for i in a[_size]:
            avg_list += list(np.diag(i))


        std = np.std(avg_list)

        if "Partial" in f:
            pType = f[:-3].split('_')
            if "1" in f:
                pType = '_'.join([pType[0], 'Audio', pType[1]])
            else: 
                pType =  '_'.join([pType[0], 'Lyric', pType[1]])
            writef.write(pType + ', ' +str(std) +  '\n')
        else:
            writef.write(f[:-2] + ', ' +str(std) + '\n')

writef.close()