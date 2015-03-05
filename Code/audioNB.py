import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import hdf5_getters as hdf
import os
import pprint 


DATA_PATH = '/Volumes/My Passport for Mac/FinalDataset/'
EMOTION_LIST = ['happy', 'sad','angry']
SEG_NUMB = 100
TEST_SAMPLES = 100
new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "


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
	results = []
	classes = []
	size_em = len(EMOTION_LIST)
	total_songs = TEST_SAMPLES * size_em

	res = clf.predict(test_set)

	for song in range(total_songs):
		classification = res[song * SEG_NUMB: song*SEG_NUMB + SEG_NUMB]
		class_counter = Counter(classification)

		if len(class_counter.most_common()) > 0:
			results.append(class_counter.most_common(1)[0][0])
			classes.append(EMOTION_LIST[song / TEST_SAMPLES])
	return results, classes


def create_sample_dict(training_set):
	sampled_dict = dict()
	# Creating the base numpy arrays 
	for em in EMOTION_LIST:	
		# Reading and sampling filenames 
		entire_lst = os.listdir(DATA_PATH + em + '/')
		numb = len(entire_lst)
		indx = random_index(numb-1, training_set + TEST_SAMPLES)
		# What will be used 
		sampled_dict[em] = np.array( entire_lst )[indx]
		# Clearing to avoid carrying around the large list -- not pythonic I know

	return sampled_dict




def extract_features(sampled_dict, training_set):
	feature_vector = [None]
	class_vector = []
	test_vector = [None]
	test_class = []

	for em in EMOTION_LIST:
		count = 0 

		for fname in sampled_dict[em]:
			f = read_hdf5(DATA_PATH + em + '/' +  fname)
			if not f:
				continue

			# Extract and sample the timbre 
			seg = hdf.get_segments_timbre(f)
			seg_indx = random_index(seg.shape[0]-1, SEG_NUMB)
			sampled_seg = seg[seg_indx,:]

			count += 1

			if count > training_set:
				# It is testing data
				if None in test_vector:
					test_vector = sampled_seg
				else:
					test_vector = np.concatenate((test_vector,sampled_seg))

				test_class += [em] * SEG_NUMB

			else:
				# It is training data
				if None in feature_vector:
					feature_vector = sampled_seg
				else:
					feature_vector = np.concatenate((feature_vector,sampled_seg))
				
				class_vector += [em] * SEG_NUMB
			
			f.close()

	return (feature_vector, class_vector, test_vector, test_class)

def post_process_results(results_arr):
	# Summarizing data

	predict = results_arr[0]
	true = results_arr[1]
	size_em = len(EMOTION_LIST)
	total_songs = TEST_SAMPLES * size_em
	correct = 0.0

	for song_i in range(len(predict)):
		song_cls = predict[song_i]
		true_cls = true[song_i]

		if song_cls in true_cls:
			correct += 1

	return correct/total_songs




def simple_svm_train(training_set):

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



training_set = [50, 100, 500, 1000]
result_dict = dict()
		
running = 0 	



for _size in training_set:
	result_array = []

	print new_training_string + str(_size)

	for count in range(50):
		print count
		running += 1
		result_array.append(simple_svm_train(_size))
	result_dict[_size] = result_array


for _sz in result_dict:
	ar = np.array(result_dict[_sz])
	result_dict[_sz] = (np.average(ar), np.std(ar), np.max(ar), np.min(ar))


pprint.pprint(result_dict)




f = open('resAudio.txt','w')
f.write('Results:\n')
for r in training_set:
	f.write(str(r) + ': ' + str(np.average(result_dict[r])) + '\n')
for r in training_set:
	f.write(str(r) + ': ' + str(result_dict[r])+ '\n')
f.close()




