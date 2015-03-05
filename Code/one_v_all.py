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
TEST_SAMPLES = 20
new_training_string = "************* NEW NUMBER OF ITERATIONS **********"


def random_index(max_val, amount):
	return np.random.random_integers(0,max_val,(amount,))

def read_hdf5(file_path):
	try:
		return hdf.open_h5_file_read(file_path)
	except:
		tok = file_path.split('/')
		print 'Skipped: ' + tok[-1]
		return None

def test_classifier(clf, emotion, test_set):
	results = dict()

	for em in test_set:
		results[em] = clf.predict(test_set[em])

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




def extract_features(sampled_dict, emotion, training_set):
	feature_vector = [None]
	class_vector = []
	test_vector = dict()
	test_class = []

	

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
				else:
					test_vector[em] = np.concatenate((test_vector[em],sampled_seg))
				test_class += [emotion in em] * SEG_NUMB

			else:
				# It is training data
				if None in feature_vector:
					feature_vector = sampled_seg	
				else:
					feature_vector = np.concatenate((feature_vector,sampled_seg))
				class_vector += [emotion in em] * SEG_NUMB
			
			f.close()
			
	# feature_vector = preprocessing.MinMaxScaler().fit_transform(feature_vector)
	# test_vector    = preprocessing.MinMaxScaler().fit_transform(test_vector)

	return (feature_vector, class_vector, test_vector, test_class)

def post_process_results(results_arr, emotion):
	# Summarizing data

	results = dict()

	confusion_matrix = np.array([0, 0, 0, 0])
	for em in results_arr:
		emo_test = [] 

		# print emotion + ' vs ' + em

		for i in range(TEST_SAMPLES):

			splice = results_arr[em][i*SEG_NUMB:i*SEG_NUMB + SEG_NUMB]
			res_count = np.bincount(splice, minlength = 2)
						
			wasCorrect = (res_count[0] < res_count[1]) == (emotion in em)

			indx = 2*int(emotion in em) + int(res_count[0] < res_count[1])

			confusion_matrix[indx] += 1 

			# print 'Song ' + str(i) + ': ' + str(res_count) + ' ' + str(wasCorrect)


			emo_test.append(wasCorrect)

		results[em] = emo_test
		# print (em,res_count[indx],len(results_arr[em]), res_count)

	print 'Correct False | Incorrect True | Incorrect False | Correct True '
	print confusion_matrix
	print (100 * confusion_matrix /  np.array([(len(EMOTION_LIST)-1)*TEST_SAMPLES, 1e10, 1e10, TEST_SAMPLES])).astype(int)
	# pprint.pprint(results)
	# results = count*1.0/(run_sum*1.0)



	count = 0 
	run_sum = 0

	for k in results:
		count += sum(results[k])
		run_sum += len(results[k])

	return count*1.0/(run_sum*1.0)




def simple_svm_train(emotion, training_set):

	song_list = []
	sizes_list = []
	other_emotions = []

	# print 'Start to sample set'
	# Setting up the data
	sampled_dict = create_sample_dict(training_set)
	# print 'Set sampled, extracting features'
	feature_vector, class_vector, test_values, test_class = extract_features(sampled_dict, emotion, training_set)

	# Creating the classifier using sklearn
	# print 'Extracted features, training classifier'
	clf = GaussianNB()
	clf.fit(feature_vector,class_vector)

	# clf = svm.SVC(max_iter = 10000)
	# clf.fit(feature_vector,class_vector)
	# print 'Finished training classifier'


	# Testing and analyzing results
	results = test_classifier(clf, emotion, test_values)
	return  post_process_results(results, emotion)




# simple_svm_train('sad',20)

training_set = [1000]
result_dict = dict()
		
running = 0 	



for iterations in training_set:
	result_array = []
	print new_training_string
	for count in range(1):
		for emo in EMOTION_LIST:
			print 'Testing: ' + emo + ' at ' +  str(iterations)
			running += 1
			result_array.append(simple_svm_train(emo, iterations))
	result_dict[iterations] = result_array


for it in result_dict:
	ar = np.array(result_dict[it])
	result_dict[it] = np.average(ar)


pprint.pprint(result_dict)




# f = open('res.txt','w+')
# f.write('Results:\n')
# for r in result_dict:
# 	f.write(str(r) + ': ' + str(np.average(result_dict[r])) + '\n')
# for r in result_dict:
# 	f.write(str(r) + ': ' + str(result_dict[r])+ '\n')
# f.close()




