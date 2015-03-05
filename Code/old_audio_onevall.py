import numpy as np
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
	results = dict()

	for group in clf:
		results[group] = dict()
		for em in test_set:
			results[group][em] = clf[group].predict(test_set[em])

	return results

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
	class_vector = dict()
	test_vector = dict()
	test_class = dict()

	for emotion in EMOTION_LIST:
		class_vector[emotion] = []
		test_class[emotion] = []
	

	for em in EMOTION_LIST:

		count = 0 
		for fname in sampled_dict[em]:

			f = read_hdf5(DATA_PATH + em + '/' +  fname)
			if not f:
				continue

			# Exctract and sample the timbre 
			seg = hdf.get_segments_timbre(f)
			seg_indx = random_index(seg.shape[0]-1, SEG_NUMB)
			sampled_seg = seg[seg_indx,:]

			count += 1

			if count > training_set:
				# It is testing data
				if em not in test_vector:
					test_vector[em] = sampled_seg
					test_class[em] = []
				else:
					test_vector[em] = np.concatenate((test_vector[em],sampled_seg))

				for emotion in EMOTION_LIST:
					test_class[emotion] += [emotion in em] * SEG_NUMB

			else:
				# It is training data
				if None in feature_vector:
					feature_vector = sampled_seg
					class_vector[em] = []	
				else:
					feature_vector = np.concatenate((feature_vector,sampled_seg))
				
				for emotion in EMOTION_LIST:
					class_vector[emotion] += [emotion in em] * SEG_NUMB
			
			f.close()

	# feature_vector = preprocessing.MinMaxScaler().fit_transform(feature_vector)
	# test_vector    = preprocessing.MinMaxScaler().fit_transform(test_vector)

	return (feature_vector, class_vector, test_vector, test_class)

def post_process_results(results_arr):
	# Summarizing data

	size_em = len(EMOTION_LIST)
	results = np.zeros((size_em,size_em,TEST_SAMPLES))

	# print results


	for i_em in range(size_em):

		for i_song_em in range(size_em):

			for i_song in range(TEST_SAMPLES):

				em_cls = EMOTION_LIST[i_em]
				em_song = EMOTION_LIST[i_song_em]

				classification = results_arr[em_cls][em_song]
				results[i_em,i_song_em,i_song] = sum(classification[i_song*SEG_NUMB:i_song*SEG_NUMB + SEG_NUMB])


			# results[i_em,i_song]  = sum(cls_used[em_song][i_song*SEG_NUMB:i_song*SEG_NUMB + SEG_NUMB])

	count = 0.0

	for i in range(TEST_SAMPLES):
		for check in EMOTION_LIST:

			choice_vector = results[:,EMOTION_LIST.index(check),i]
			max_indx = np.argmax(choice_vector)

			if max_indx.size > 1:
				print 'Random Choice'
				print choice_vector
				max_indx = np.random.choice(max_indx)

			choice = EMOTION_LIST[max_indx]
			correct =   float(check in choice)
			count += correct

	return count / (TEST_SAMPLES * size_em * 1.0)




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

	clf_d = dict()

	for em in EMOTION_LIST:
		clf_d[em] = GaussianNB()
		clf_d[em].fit(feature_vector,class_vector[em])

	# print 'Finished training classifier'


	# Testing and analyzing results
	results = test_classifier(clf_d, test_values)
	return  post_process_results(results)



training_set = [20, 50, 100, 250, 500, 1000, 1500]
result_dict = dict()
		
running = 0 	



for _size in training_set:
	result_array = []

	print new_training_string + str(_size)

	for count in range(10):
		running += 1
		result_array.append(simple_svm_train(_size))
	result_dict[_size] = result_array


# for _sz in result_dict:
# 	ar = np.array(result_dict[_sz])
# 	result_dict[_sz] = np.average(ar)


pprint.pprint(result_dict)




f = open('resAudio.txt','w+')
f.write('Results:\n')
for r in result_dict:
	f.write(str(r) + ': ' + str(np.average(result_dict[r])) + '\n')
for r in result_dict:
	f.write(str(r) + ': ' + str(result_dict[r])+ '\n')
f.close()




