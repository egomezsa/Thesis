import numpy as np
import sys
import pprint
import os
import _MultimodalTools as MT
from collections import Counter
import pickle

TRAINING_COUNT = 100
TESTING_COUNT = 300
EMOTION_LIST = ['happy', 'sad','angry']

i_dict = {'feature_train': 0, 'class_train': 1, 'feature_test': 2, 'class_test' : 3}

new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "

def create_features(tst, tr_count = TRAINING_COUNT, tst_count = TESTING_COUNT):
	sample_dict = MT.create_sample_dict(tr_count, tst_count)
	bow_tup = MT.extract_features_BOW(sample_dict, tr_count, tst_count, tst)
	au_tup = MT.extract_features_Audio(sample_dict, tr_count, tst_count, tst)
	return ((bow_tup,au_tup))

def build_ensemble(bow, au, inputstr, tr_count = TRAINING_COUNT):
	ft = i_dict['feature_train']
	ct = i_dict['class_train']


	clf_au  = MT.get_Classifier(inputstr[0])(au[ft], au[ct])
	clf_bow = MT.get_Classifier(inputstr[1])(bow[ft], bow[ct])

	combined_vector = MT.post_process_audio(clf_au.predict(au[ft]))
	res_bow = clf_bow.predict(bow[ft])

	for i in range(len(combined_vector)):
		combined_vector[i].append(EMOTION_LIST.index(res_bow[i]))

	ensemble_clf =  MT.get_Classifier(inputstr[2])(combined_vector,bow[ct])

	return (ensemble_clf,clf_bow, clf_au)

def run_ensemble(clf_tuple, bow, au, tst_count = TESTING_COUNT):

	ft = i_dict['feature_test']
	ct = i_dict['class_test']

	clf_en = clf_tuple[0]
	clf_bow = clf_tuple[1]
	clf_au = clf_tuple[2]


	combined_vector = MT.post_process_audio(clf_au.predict(au[ft]))
	res_bow = clf_bow.predict(bow[ft])

	for i in range(len(combined_vector)):
		combined_vector[i].append(EMOTION_LIST.index(res_bow[i]))


	res =  clf_en.predict(combined_vector)
	return res


def ensemble_test(train_size, inputstr):

	bow, au = create_features(inputstr, tr_count = train_size)
	ensemble_clf = build_ensemble(bow,au, inputstr, tr_count = train_size)
	res =  run_ensemble(ensemble_clf,bow,au)

	true_arr = bow[i_dict['class_test']]
	correct_ma = np.zeros((3,3))

	for r_i in range(len(res)):

		pred_indx = EMOTION_LIST.index(res[r_i])
		true_indx = EMOTION_LIST.index(true_arr[r_i])

		correct_ma[pred_indx,true_indx] += 1


	print correct_ma
	return  correct_ma


tst = 'MMM'

if len(sys.argv) < 2:
	inputstr = './run/Ensemble.p'
else:
	tst = sys.argv[1]
	inputstr = './run/Ensemble_'  + sys.argv[1] + '.p'


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
			size_dict[_size].append(ensemble_test(_size, tst))
			pickle.dump(size_dict, open(inputstr, 'w'))

	else:
		size_dict[_size] = []

		for count in range(30):
			print count
			size_dict[_size].append(ensemble_test(_size, tst))
			pickle.dump(size_dict, open(inputstr, 'w'))


print 'done'
