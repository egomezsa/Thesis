import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import hdf5_getters as hdf
import os
import pickle
import pprint 


DATA_PATH_ExHD = '/Volumes/My Passport for Mac/FinalDataset/'
DATA_PATH_BOW = '../pfiles/'
EMOTION_LIST = ['happy', 'sad','angry']
SEG_NUMB_AUD = 100
VECTOR_SIZE_BOW = 5000
new_training_string = "************* NEW NUMBER OF ITERATIONS **********  "


def random_index(max_val, amount):
	return np.random.random_integers(0,max_val,(amount,))

def read_hdf5(file_path):
	try:
		return hdf.open_h5_file_read(file_path)
	except:
		print file_path
		tok = file_path.split('/')
		print 'Skipped: ' + tok[-1]
		return None

def create_sample_dict(training_count, testing_count):
	sampled_dict = dict()
	# Creating the base numpy arrays 
	for em in EMOTION_LIST:	
		# Reading and sampling filenames 
		entire_lst = pickle.load(open(DATA_PATH_BOW+ em + '_listing.p','r'))

		numb = len(entire_lst)
		indx = random_index(numb-1, training_count + testing_count)
		# # What will be used 
		sampled_dict[em] = np.array( entire_lst )[indx]

	return sampled_dict


def extract_features_BOW(sampled_dict, training_count, testing_count):
	
	feature_vector = np.zeros((len(EMOTION_LIST)*training_count, VECTOR_SIZE_BOW))
	test_vector = np.zeros((len(EMOTION_LIST)*testing_count, VECTOR_SIZE_BOW))
	class_vector = []
	test_class = []
	
	classes = dict()

	for em in EMOTION_LIST:

		emotion_dict = pickle.load(open(DATA_PATH_BOW + em + '_BOW.p','r'))
		classes[em] = []

		vectors = np.zeros((training_count+testing_count,VECTOR_SIZE_BOW))
		count = 0

		for song in sampled_dict[em]:
			v = np.zeros((VECTOR_SIZE_BOW,))
			for m in emotion_dict[song]:
				if ':' in m:
					tok = m.split(':')
					indx = int(tok[0])
					wordcount = int(tok[1])
					if int(tok[0]) < VECTOR_SIZE_BOW:
						v[indx] = wordcount
			vectors[count,:] = v
			classes[em] += [em]
			count += 1

		indx = EMOTION_LIST.index(em)
		offset_tr = training_count * indx
		offset_tst = testing_count * indx

		feature_vector[0 + offset_tr : training_count + offset_tr,:] = vectors[0:training_count,:]
		test_vector[0 + offset_tst :testing_count + offset_tst,:] = vectors[training_count:,:]
		class_vector += [em] * training_count
		test_class  += [em] * testing_count

	return (feature_vector, class_vector, test_vector, test_class)

def extract_features_Audio(sampled_dict, training_count, testing_count, seg_numb = SEG_NUMB_AUD):
	feature_vector = [None]
	class_vector = []
	test_vector = [None]
	test_class = []

	for em in EMOTION_LIST:
		count = 0 

		for fname in sampled_dict[em]:
			f = read_hdf5(DATA_PATH_ExHD + em + '/' +  fname +'.h5')
			if not f:
				continue

			# Extract and sample the timbre 
			seg = hdf.get_segments_timbre(f)
			seg_indx = random_index(seg.shape[0]-1, seg_numb)
			sampled_seg = seg[seg_indx,:]

			count += 1

			if count > training_count:
				# It is testing data
				if None in test_vector:
					test_vector = sampled_seg
				else:
					test_vector = np.concatenate((test_vector,sampled_seg))

				test_class += [em] * seg_numb

			else:
				# It is training data
				if None in feature_vector:
					feature_vector = sampled_seg
				else:
					feature_vector = np.concatenate((feature_vector,sampled_seg))
				
				class_vector += [em] * seg_numb
			
			f.close()

	return (feature_vector, class_vector, test_vector, test_class)

def post_process_audio(results_arr, test_count, seg_numb = SEG_NUMB_AUD):
	# Summarizing data

	size_em = len(EMOTION_LIST)
	total_songs = test_count * size_em
	results = []

	for song in range(total_songs):
		classification = results_arr[song * seg_numb: song*seg_numb + seg_numb]
		class_counter = Counter(classification)
		lst = [] 

		for em in EMOTION_LIST:
			if em in class_counter:
				lst.append(class_counter[em])
			else:
				lst.append(0)

		results.append(lst)

	return results


def simple_GNB_train(feature_vector, class_vector):
	clf = GaussianNB()
	clf.fit(feature_vector,class_vector)
	return  clf



