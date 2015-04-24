import numpy as np
import sys
import pprint
import os
import _MultimodalTools as MT
from collections import Counter
import pickle

TRAINING_COUNT = 100
TESTING_COUNT = 100
EMOTION_LIST = ['happy', 'sad','angry']

i_dict = {'feature_train': 0, 'class_train': 1, 'feature_test': 2, 'class_test' : 3}

new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "

def create_features(tst, tr_count = TRAINING_COUNT, tst_count = TESTING_COUNT):
	sample_dict = MT.create_sample_dict(tr_count, tst_count)
	au_tup = MT.extract_features_Audio(sample_dict, tr_count, tst_count, tst, normalize = 'M' in tst)
	bow_tup = MT.extract_features_BOW(sample_dict, tr_count, tst_count, tst)
	return ((bow_tup,au_tup))

def build_ensemble(bow, au,  tst, tr_count = TRAINING_COUNT):
	ft = i_dict['feature_train']
	ct = i_dict['class_train']


	mode = int(tst[2])

	ensemble_clf = None
	other_clf = None

	print 'Mode: ' + str(mode)

	if mode == 1 : 

		other_clf  = MT.get_Classifier(tst[0])(au[ft], au[ct],'linear')

		audio_post = MT.post_process_audio(other_clf.predict(au[ft]))

		combined_vector = np.zeros((len(audio_post),len(EMOTION_LIST) + bow[ft].shape[1]))

		for i in range(len(audio_post)):
			combined_vector[i,:] = np.concatenate((np.array(audio_post[i]), bow[ft][i]))


		ensemble_clf =  MT.get_Classifier(tst[1])(combined_vector,bow[ct])

	else:

		other_clf = MT.get_Classifier(tst[0])(bow[ft], bow[ct])
		res_bow = np.array(other_clf.predict_proba(bow[ft]))

		combined_vector = np.zeros(( au[ft].shape[0], len(EMOTION_LIST) + au[ft].shape[1] ))
		
		segment_size = au[ft].shape[0] / bow[ft].shape[0]

		for i in range(res_bow.shape[0]):
			sample_range = np.arange(i*segment_size, i*segment_size+segment_size)
			au_seg = au[ft][sample_range,:]
			bow_temp =  np.repeat(res_bow[i,:][:,None],segment_size,axis=1).T
			combined_vector[sample_range,:] = np.concatenate( (au_seg,bow_temp), axis = 1)


		ensemble_clf = MT.get_Classifier(tst[1])(combined_vector, au[ct],'linear')


	return (ensemble_clf, other_clf)

def run_ensemble(clf_tuple, bow, au, tst_count = TESTING_COUNT):

	ft = i_dict['feature_test']
	ct = i_dict['class_test']

	clf_en = clf_tuple[0]
	clf_other = clf_tuple[1]


	mode = int(tst[2])

	combined_vector = None

	if mode == 1 :

		audio_post = MT.post_process_audio(clf_other.predict(au[ft]))

		combined_vector = np.zeros((len(audio_post),len(EMOTION_LIST) + bow[ft].shape[1]))

		for i in range(len(audio_post)):
			combined_vector[i,:] = np.concatenate((np.array(audio_post[i]), bow[ft][i]))

		return clf_en.predict(combined_vector)

	else: 

		res_bow = np.array(clf_other.predict_proba(bow[ft]))
		combined_vector = np.zeros(( au[ft].shape[0], len(EMOTION_LIST) + au[ft].shape[1] ))
		
		segment_size = au[ft].shape[0] / bow[ft].shape[0]

		for i in range(res_bow.shape[0]):
			sample_range = np.arange(i*segment_size, i*segment_size+segment_size)
			au_seg = au[ft][sample_range,:]
			bow_temp =  np.repeat(res_bow[i,:][:,None],segment_size,axis=1).T
			combined_vector[sample_range,:] = np.concatenate( (au_seg,bow_temp), axis = 1)

		res =  clf_en.predict(combined_vector)
		
		predictions = [] 


		for i in range(0, len(res), segment_size):
			count = Counter(res[i:i+segment_size])		
			if len(count.most_common()) > 0:
				predictions.append(count.most_common(1)[0][0])	

		return predictions


def ensemble_test(train_size, tst):
	bow, au = create_features(tst, tr_count = train_size)
	ensemble_clf = build_ensemble(bow,au, tst, tr_count = train_size)
	res =  run_ensemble(ensemble_clf,bow,au)




	true_arr = bow[i_dict['class_test']]
	correct_ma = np.zeros((3,3))

	count = 0 
	runsum = 0 

	for r_i in range(len(res)):

		if 'None' in (res[r_i]) or 'None' in true_arr[r_i]:
			print 'skipped'
			continue

		pred_indx = EMOTION_LIST.index(res[r_i])
		true_indx = EMOTION_LIST.index(true_arr[r_i])
		correct_ma[pred_indx,true_indx] += 1



	print correct_ma	

	return  correct_ma


tst = 'MM1'

if len(sys.argv) < 2:
	inputstr = './run/PartialEnsemble.p'
else:
	tst = sys.argv[1]
	inputstr = './run/PartialEnsemble_'  + sys.argv[1] + '.p'

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

		for count in range(run_num):
			print count
			size_dict[_size].append(ensemble_test(_size, tst))
			pickle.dump(size_dict, open(inputstr, 'w'))

print 'done'