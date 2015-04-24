import numpy as np
import sys
import pprint
import os
import _MultimodalTools as MT
from collections import Counter
import datetime
import pickle

TRAINING_COUNT = 100
TESTING_COUNT = 100
SEG_NUMB_AUD = 50
EMOTION_LIST = ['happy', 'sad','angry']

i_dict = {'feature_train': 0, 'class_train': 1, 'feature_test': 2, 'class_test' : 3}

new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "

def create_features(tst, tr_count = TRAINING_COUNT, tst_count = TESTING_COUNT):
	sample_dict = MT.create_sample_dict(tr_count, tst_count)
	au_tup = MT.extract_features_Audio(sample_dict, tr_count, tst_count, tst, seg_numb = SEG_NUMB_AUD, normalize= 'M' in tst)
	bow_tup = MT.extract_features_BOW(sample_dict, tr_count, tst_count, tst)

	trn = i_dict['feature_train']
	tst = i_dict['feature_test']

	trn_clss = au_tup[i_dict['class_train']]
	tst_clss = bow_tup[i_dict['class_test']]
	vect_trn = np.zeros((au_tup[trn].shape[0], au_tup[trn].shape[1] + bow_tup[trn].shape[1]))
	vect_tst = np.zeros((au_tup[tst].shape[0], au_tup[tst].shape[1] + bow_tup[tst].shape[1]))

	for m in range(au_tup[trn].shape[0]):
		song_indx = m / (SEG_NUMB_AUD)
		vect_trn[m, :] = np.concatenate((au_tup[trn][m],bow_tup[trn][song_indx]))

	for m in range(au_tup[tst].shape[0]):
		song_indx = m / (SEG_NUMB_AUD)
		vect_tst[m, :] = np.concatenate((au_tup[tst][m],bow_tup[tst][song_indx]))


	return ((vect_trn, trn_clss, vect_tst, tst_clss))

def build_series(vects, tst, tr_count = TRAINING_COUNT):
	ft = i_dict['feature_train']
	ct = i_dict['class_train']

	clf = MT.get_Classifier(tst[0])(vects[ft], vects[ct], kernel_type="linear")

	return clf

def run_series(clf, vects, tst_count = TESTING_COUNT):

	ft = i_dict['feature_test']
	ct = i_dict['class_test']

	res = clf.predict(vects[ft])
	results = []
	size_em = len(EMOTION_LIST)
	total_songs = TESTING_COUNT * size_em

	for song in range(total_songs):
		classification = res[song * SEG_NUMB_AUD: song*SEG_NUMB_AUD + SEG_NUMB_AUD]
		class_counter = Counter(classification)

		if len(class_counter.most_common()) > 0:
			results.append(class_counter.most_common(1)[0][0])


	return results


def series_test(train_size, tst):
	print datetime.datetime.now().time()
	vects = create_features(tst, tr_count = train_size)
	series_clf = build_series(vects,  tst, tr_count = train_size)
	res =  run_series(series_clf,vects)



	true_arr = vects[i_dict['class_test']]
	correct_ma = np.zeros((3,3))

	for r_i in range(len(res)):

		if 'None' in (res[r_i]) or 'None' in true_arr[r_i]:
			print 'skipped'
			continue

		pred_indx = EMOTION_LIST.index(res[r_i])
		true_indx = EMOTION_LIST.index(true_arr[r_i])

		correct_ma[pred_indx,true_indx] += 1

	print correct_ma
	print datetime.datetime.now().time()
	return  correct_ma



tst = 'M'

if len(sys.argv) < 2:
	inputstr = './run/Series.p'
else:
	tst = sys.argv[1]
	inputstr = './run/Series_'  + sys.argv[1] + '.p'

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
			size_dict[_size].append(series_test(_size, tst))
			pickle.dump(size_dict, open(inputstr, 'w'))

	else:
		size_dict[_size] = []

		for count in range(run_num):
			print count
			size_dict[_size].append(series_test(_size, tst))
			pickle.dump(size_dict, open(inputstr, 'w'))


print 'done'


