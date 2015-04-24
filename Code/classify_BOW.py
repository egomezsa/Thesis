import numpy as np
import pickle
from collections import Counter
import _MultimodalTools as MT
import os
import sys
import pprint 

DATA_PATH = '../pfiles/'
EMOTION_LIST = ['happy', 'sad','angry']
VECTOR_SIZE = 5000
TEST_SAMPLES = 100
new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "

i_dict = {'feature_train': 0, 'class_train': 1, 'feature_test': 2, 'class_test' : 3}


def test_classifier(clf, test_set):
	results = clf.predict(test_set)
	return results

def post_process_results(results_arr, true_arr):
	# Summarizing data
	accr = []

	correct_ma = np.zeros((3,3))

	for indx in range(len(results_arr)):

		if 'None' in (results_arr[indx]) or 'None' in true_arr[indx]:
			print 'skipped'
			continue
		
		true_indx = EMOTION_LIST.index(true_arr[indx])
		pred_indx = EMOTION_LIST.index(results_arr[indx])

		correct_ma[pred_indx,true_indx] += 1


	print correct_ma	
	return correct_ma




def simple_train(training_set, tst):

	# Setting up the data
	sample_dict = MT.create_sample_dict(training_set, TEST_SAMPLES)

	# print 'Set sampled, extracting features'

	bow_features = MT.extract_features_BOW(sample_dict, training_set, TEST_SAMPLES, tst)

	feature_vector = bow_features[i_dict['feature_train']]
	class_vector = bow_features[i_dict['class_train']]
	test_values = bow_features[i_dict['feature_test']]
	test_class = bow_features[i_dict['class_test']]

	clf = MT.get_Classifier(tst[0])(feature_vector, class_vector)

	# print 'Finished training classifier'

	# Testing and analyzing results
	results = test_classifier(clf, test_values)
	return  post_process_results(results, test_class)


tst = 'M'

if len(sys.argv) < 2:
	inputstr = './run/BOW.p'
else:
	tst = sys.argv[1]
	inputstr = './run/BOW_'  + sys.argv[1] + '.p'


training_set = [20, 50, 100, 500]


if os.path.exists(inputstr):
	size_dict = pickle.load(open(inputstr,'r'))
else:
	size_dict = dict()

run_num = 30
		
for _size in training_set:

	print new_training_string + str(_size)
	print tst

	if _size in size_dict:
		for count in range(len(size_dict[_size]) , run_num):
			print count
			size_dict[_size].append(simple_train(_size, tst))
			pickle.dump(size_dict, open(inputstr, 'w'))

	else:
		size_dict[_size] = []

		for count in range(30):
			print count
			size_dict[_size].append(simple_train(_size, tst))
			pickle.dump(size_dict, open(inputstr, 'w'))


print 'done'

