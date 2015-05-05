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

        sens_ma = np.zeros((3,len(a[_size])))

        for i in range(len(a[_size])):

            v = a[_size][i]



            lst = []

            for j in range(3):

                tp =  v[j,j]
                fn = (np.sum(v) + v[j,j] - np.sum(np.diag(v)) - np.sum(v[j,:]))

                tn = np.sum(np.diag(v)) - v[j,j]
                fp = np.sum(v[j,:]) - v[j,j]

                # print tn,fp,(tn/(tn+fp))
                lst.append(tp/(tp + fn))
                # lst.append(tn/(tn + fp))
            sens_ma[:,i] = np.array(lst)
        # print sens_ma
        avgs =  np.average(sens_ma)
        # exit()

        # for i in a[_size]:
        #     avg_list += list(np.diag(i))


        # std = np.std(avg_list)

        if "Partial" in f:
            pType = f[:-3].split('_')
            if "1" in f:
                pType = '_'.join([pType[0], 'Audio', pType[1]])
            else: 
                pType =  '_'.join([pType[0], 'Lyric', pType[1]])
            writef.write(pType + ', ' +str(avgs) + '\n')#+  ', ' + str(avgs[1]) + ', ' + str(avgs[2]) + '\n')
        elif "BOW" in f:
            writef.write(f[:-2] + ', ' +str(avgs)+ '\n')# +  ', ' + str(avgs[1]) + ', ' + str(avgs[2]) + '\n')
        elif "Ensemble" in f:
            writef.write('F' + f[:-2] + ', ' +str(avgs)+ '\n')# +  ', ' + str(avgs[1]) + ', ' + str(avgs[2]) + '\n')
        else:
            writef.write(f[:-2] + ', ' +str(avgs) + '\n')#+  ', ' + str(avgs[1]) + ', ' + str(avgs[2]) + '\n')

writef.close()
