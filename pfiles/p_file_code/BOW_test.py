import pickle

f = open('../mxm_dataset_train.txt', 'r')
out = dict()

for line in f:
	line = line[:-1]
	if not (line.startswith('%') or line.startswith('#')):
		toks = line.split(',')
		out[toks[0]] = toks[1:]
f.close()

# f = open('../mxm_dataset_train.txt', 'r')
# for line in f:
# 	line = line[:-1]
# 	if not (line.startswith('%') or line.startswith('#')):
# 		toks = line.split(',')
# 		out[toks[0]] = toks[1:]
# f.close()


pickle.dump(out, open('../preproc_BOW.p', 'w+'))

