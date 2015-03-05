import pickle
import pprint

pre_path = '../pfiles/'
sorted_files = ['happy','sad','angry']
vector_dict = pickle.load(open('../preproc_BOW.p', 'r'))

print 'File loaded'

reordered = dict()


for p in sorted_files:

	p_dict = pickle.load(open(pre_path + p + '.p', 'r'))

	f = open(pre_path + p + '_BOW.p', 'r')
	reordered[p] = pickle.load(f)
	f.close()

	for f in p_dict:

		if f in vector_dict and f not in reordered[p]:
			reordered[p][f] = vector_dict[f]



# for p in sorted_files:
# 	f = open(pre_path + p + '_BOW.p', 'w+')
# 	pickle.dump(reordered[p],f)
# 	print len(reordered[p])
# 	f.close()

