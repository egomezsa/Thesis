import numpy as np
import pickle
from sklearn import svm
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import os
import pprint 

DATA_PATH = '../pfiles/'
EMOTION_LIST = ['happy', 'sad','angry']
VECTOR_SIZE = 5000
TEST_SAMPLES = 20
new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "
vector_mask = pickle.load(open('../NRCLexicon.p','r'))
	

def random_index(max_val, amount):
	return np.random.random_integers(0,max_val,(amount,))

def read_hdf5(file_path):
	try:
		return hdf.open_h5_file_read(file_path)
	except:
		tok = file_path.split('/')
		print 'Skipped: ' + tok[-1]
		return None

def test_classifier(clf, test_set):
	results = clf.predict(test_set)
	return results

def create_sample_dict(training_set):
	sampled_dict = dict()
	# Creating the base numpy arrays 
	for em in EMOTION_LIST:	
		# Reading and sampling filenames 
		entire_lst = pickle.load(open(DATA_PATH+ em + '_listing.p','r'))
		numb = len(entire_lst)
		indx = random_index(numb-1, training_set + TEST_SAMPLES)
		# # What will be used 
		sampled_dict[em] = np.array( entire_lst )[indx]

	return sampled_dict




def extract_features(sampled_dict, training_set):
	
	feature_vector = np.zeros((len(EMOTION_LIST)*training_set, 3*VECTOR_SIZE))
	test_vector = np.zeros((len(EMOTION_LIST)*TEST_SAMPLES, 3*VECTOR_SIZE))
	class_vector = []
	test_class = []
	
	classes = dict()

	count_correct = 0
	count_total = 0 

	for em in EMOTION_LIST:

		emotion_dict = pickle.load(open(DATA_PATH + em + '_BOW.p','r'))
		classes[em] = []

		vectors = np.zeros((training_set+TEST_SAMPLES, 3*VECTOR_SIZE))
		count = 0

		for song in sampled_dict[em]:
			v = np.zeros((VECTOR_SIZE,))
			for m in emotion_dict[song]:
				if ':' in m:
					tok = m.split(':')
					indx = int(tok[0])
					wordcount = int(tok[1])
					if int(tok[0]) < VECTOR_SIZE:
						v[indx] = wordcount
			vectors[count,:] = (v + 10*np.multiply(vector_mask,v)).flatten()
			classes[em] += [em]
			count += 1

		indx = EMOTION_LIST.index(em)
		offset_tr = training_set * indx
		offset_tst = TEST_SAMPLES * indx

		feature_vector[0 + offset_tr : training_set + offset_tr,:] = vectors[0:training_set,:]
		test_vector[0 + offset_tst :TEST_SAMPLES + offset_tst,:] = vectors[training_set:,:]
		class_vector += [em] * training_set
		test_class  += [em] * TEST_SAMPLES

	return (feature_vector, class_vector, test_vector, test_class)

def post_process_results(results_arr):
	# Summarizing data
	accr = []

	for indx in range(len(EMOTION_LIST)):
		em = EMOTION_LIST[indx]
		offset = indx * TEST_SAMPLES
		class_counter = Counter(results_arr[0 + offset: TEST_SAMPLES + offset])

		if em in class_counter:
			accr += [class_counter[em] * 1.0 / TEST_SAMPLES]



		

	return np.average(np.array(accr))




def simple_train(training_set):

	song_list = []
	sizes_list = []
	other_emotions = []

	# print 'Start to sample set'
	# Setting up the data
	sampled_dict = create_sample_dict(training_set)
	# print 'Set sampled, extracting features'

	feature_vector, class_vector, test_values, test_class = extract_features(sampled_dict, training_set)

	# Creating the classifier using sklearn
	# print 'Extracted features, training classifier'

	clf = GaussianNB()
	clf.fit(feature_vector,class_vector)

	# print 'Finished training classifier'

	# Testing and analyzing results
	results = test_classifier(clf, test_values)
	return  post_process_results(results)



training_set = [20, 50, 100, 500, 1000]
result_dict = dict()
		
running = 0 	


for _size in training_set:
	result_array = []

	print new_training_string + str(_size)

	for count in range(5):
		running += 1
		result_array.append(simple_train(_size))
	result_dict[_size] = result_array



# pprint.pprint(result_dict)
# for _sz in result_dict:
# 	ar = np.array(result_dict[_sz])
# 	result_dict[_sz] = np.average(ar)


# pprint.pprint(result_dict)




f = open('resBOWSupervised.txt','w')
f.write('Results:\n')
for r in training_set:
	f.write(str(r) + ': ' + str(np.average(result_dict[r])) + '\n')
for r in training_set:
	f.write(str(r) + ': ' + str(result_dict[r])+ '\n')
f.close()




