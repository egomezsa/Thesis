import numpy as np
import pprint
import _MultimodalTools as MT
from collections import Counter

TRAINING_COUNT = 100
TESTING_COUNT = 100
SEG_NUMB_AUD = 50
EMOTION_LIST = ['happy', 'sad','angry']

i_dict = {'feature_train': 0, 'class_train': 1, 'feature_test': 2, 'class_test' : 3}

new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "

def create_features(tr_count = TRAINING_COUNT, tst_count = TESTING_COUNT):
	sample_dict = MT.create_sample_dict(tr_count, tst_count)
	bow_tup = MT.extract_features_BOW(sample_dict, tr_count, tst_count)
	au_tup = MT.extract_features_Audio(sample_dict, tr_count, tst_count, seg_numb = SEG_NUMB_AUD)

	trn = i_dict['feature_train']
	tst = i_dict['feature_test']

	trn_clss = au_tup[i_dict['class_train']]
	tst_clss = au_tup[i_dict['class_test']]
	vect_trn = np.zeros((au_tup[trn].shape[0], au_tup[trn].shape[1] + bow_tup[trn].shape[1]))
	vect_tst = np.zeros((au_tup[tst].shape[0], au_tup[tst].shape[1] + bow_tup[tst].shape[1]))

	for m in range(au_tup[trn].shape[0]):
		song_indx = m / (SEG_NUMB_AUD)
		vect_trn[m, :] = np.concatenate((au_tup[trn][m],bow_tup[trn][song_indx]))

	for m in range(au_tup[tst].shape[0]):
		song_indx = m / (SEG_NUMB_AUD)
		vect_tst[m, :] = np.concatenate((au_tup[tst][m],bow_tup[tst][song_indx]))


	return ((vect_trn, trn_clss, vect_tst, tst_clss))

def build_series(vects, tr_count = TRAINING_COUNT):
	ft = i_dict['feature_train']
	ct = i_dict['class_train']

	clf = MT.simple_GNB_train(vects[ft], vects[ct])

	return clf

def run_series(clf, vects, tst_count = TESTING_COUNT):

	ft = i_dict['feature_test']
	ct = i_dict['class_test']

	res = clf.predict(vects[ft])
	results = []
	classes = []
	size_em = len(EMOTION_LIST)
	total_songs = TESTING_COUNT * size_em

	for song in range(total_songs):
		classification = res[song * SEG_NUMB_AUD: song*SEG_NUMB_AUD + SEG_NUMB_AUD]
		class_counter = Counter(classification)

		if len(class_counter.most_common()) > 0:
			results.append(class_counter.most_common(1)[0][0])
			classes.append(EMOTION_LIST[song / TESTING_COUNT])


	return results, classes


def series_test(train_size):
	vects = create_features(tr_count = train_size)
	series_clf = build_series(vects, tr_count = train_size)
	res,clss =  run_series(series_clf,vects)


	count = 0 
	runsum = 0 
	for r_i in range(len(res)):
		r = res[r_i]
		c = clss[r_i]

		runsum += 1
		if r in c:
			count += 1

	return  1.0 * count /  runsum

test_sizes = [50, 100, 500, 1000]
f = open('series.txt','w')
f.write('Results:\n')
for sz in test_sizes:
	f.write('Size: ' + str(sz) + '\n')

	lst = []
	print new_training_string + ' ' + str(sz)
	for i in range(50):
		print i
		lst.append(series_test(sz))

	f.write(str(lst) + '\n')
	f.write(str((np.average(np.array(lst)), np.std(np.array(lst)), max(lst), min(lst))) + '\n')
f.close()




