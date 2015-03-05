import pickle
import pprint

sentiment_list = ['happy', 'sad', 'angry']

for em in sentiment_list:

	target_file = '../pfiles/' + em + '.p'

	f_BOW = open('../pfiles/' + em + '_BOW.p', 'r')
	f_org = open(target_file, 'r')

	bow_dict = pickle.load(f_BOW)
	f_lst   = pickle.load(f_org)

	f_BOW.close()
	f_org.close()


	em_lst =  []

	for k in f_lst:
		if k in bow_dict:
			em_lst.append(k)



	f_tar = open(target_file,'w')
	pickle.dump(em_lst,f_tar)
	f_tar.close()




