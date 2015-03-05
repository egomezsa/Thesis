import numpy as np
import pprint
import _MultimodalTools as MT
from collections import Counter

TRAINING_COUNT = 100
TESTING_COUNT = 100
SEG_NUMB_AUD = 100
EMOTION_LIST = ['happy', 'sad','angry']

i_dict = {'feature_train': 0, 'class_train': 1, 'feature_test': 2, 'class_test' : 3}

new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "

def create_features(tr_count = TRAINING_COUNT, tst_count = TESTING_COUNT):
	sample_dict = MT.create_sample_dict(tr_count, tst_count)
	bow_tup = MT.extract_features_BOW(sample_dict, tr_count, tst_count)
	au_tup = MT.extract_features_Audio(sample_dict, tr_count, tst_count)

	trn = i_dict['feature_train']
	tst = i_dict['feature_test']

	trn_clss = au_tup[i_dict['class_train']]
	tst_clss = au_tup[i_dict['class_test']]
	vect_trn = np.zeros((au_tup[trn].shape[0], au_tup[trn].shape[1] + bow_tup[trn].shape[1]))
	vect_tst = np.zeros((au_tup[tst].shape[0], au_tup[tst].shape[1] + bow_tup[tst].shape[1]))

	for m in range(len(au_tup[trn])):
		song_indx = m / (SEG_NUMB_AUD)
		vect_trn[m, :] = np.concatenate((au_tup[trn][m],bow_tup[trn][song_indx]))

	bow_tup[trn] = []

	for m in range(len(au_tup[tst])):
		song_indx = m / (SEG_NUMB_AUD)
		vect_tst[m, :] = np.concatenate((au_tup[tst][m],bow_tup[tst][song_indx]))

	bow_tup[tst] = []

	return ((vect_trn, trn_clss, vect_tst, tst_clss))

def build_series(bow, au, tr_count = TRAINING_COUNT):
	ft = i_dict['feature_train']
	ct = i_dict['class_train']

	clf_bow = MT.simple_GNB_train(bow[ft], bow[ct])
	clf_au  = MT.simple_GNB_train(au[ft], au[ct])

	combined_vector = MT.post_process_audio(clf_au.predict(au[ft]), tr_count)
	res_bow = clf_bow.predict(bow[ft])

	for i in range(len(combined_vector)):
		combined_vector[i].append(EMOTION_LIST.index(res_bow[i]))

	ensemble_clf =  MT.simple_GNB_train(combined_vector,bow[ct])

	return (ensemble_clf,clf_bow, clf_au)

def run_series(clf_tuple, bow, au, tst_count = TESTING_COUNT):

	ft = i_dict['feature_test']
	ct = i_dict['class_test']

	clf_en = clf_tuple[0]
	clf_bow = clf_tuple[1]
	clf_au = clf_tuple[2]


	combined_vector = MT.post_process_audio(clf_au.predict(au[ft]), tst_count)
	res_bow = clf_bow.predict(bow[ft])

	for i in range(len(combined_vector)):
		combined_vector[i].append(EMOTION_LIST.index(res_bow[i]))

	return clf_en.predict(combined_vector)


def series_test(train_size):
	bow, au = create_features(tr_count = train_size)
	ensemble_clf = build_ensemble(bow,au, tr_count = train_size)
	res =  run_ensemble(ensemble_clf,bow,au)


	count = 0 
	runsum = 0 
	for r_i in range(len(res)):
		r = res[r_i]
		runsum += 1
		if r in EMOTION_LIST[r_i / TESTING_COUNT]:
			count += 1

	return  1.0 * count /  runsum

print 'hi'

test_sizes = [500]
f = open('series.txt','w')
f.write('Results:\n')
for sz in test_sizes:
	f.write('Size: ' + str(sz) + '\n')

	lst = []
	print new_training_string
	for i in range(10):
		lst.append(series_test(sz))

	f.write(str(lst) + '\n')
	f.write(str((np.average(np.array(lst)), np.std(np.array(lst)), max(lst), min(lst))) + '\n')
f.close()




