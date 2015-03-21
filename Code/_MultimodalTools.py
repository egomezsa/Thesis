# 2015.03.20 20:34:37 CST
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.metrics import pairwise
import hdf5_getters as hdf
import os
import pickle
import pprint
DATA_PATH_ExHD = '/Volumes/My Passport for Mac/FinalDataset/'
DATA_PATH_BOW = '../pfiles/'
EMOTION_LIST = ['happy', 'sad', 'angry']
SEG_NUMB_AUD = 100
VECTOR_SIZE_BOW = 5000
new_training_string = '************* NEW NUMBER OF ITERATIONS **********  '

def histogram_intersection_kernel(X, Y = None, alpha = None, beta = None):
    """
    Compute the histogram intersection kernel(min kernel) 
    between X and Y::
        K(x, y) = \\sum_i^n min(|x_i|^\x07lpha, |y_i|^\x08eta)
    Parameters
    ----------
    X : array of shape (n_samples_1, n_features)
    Y : array of shape (n_samples_2, n_features)
    gamma : float
    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    (X, Y,) = pairwise.check_pairwise_arrays(X, Y)
    if alpha is not None:
        X = np.abs(X) ** alpha
    if beta is not None:
        Y = np.abs(Y) ** beta
    (n_samples_1, n_features,) = X.shape
    (n_samples_2, _,) = Y.shape
    K = np.zeros(shape=(n_samples_1, n_samples_2), dtype=np.float)
    for i in range(n_samples_1):
        K[i] = np.sum(np.minimum(X[i], Y), axis=1)

    return K



def random_index(max_val, amount):
    return np.random.random_integers(0, max_val, (amount,))



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
    for em in EMOTION_LIST:
        entire_lst = pickle.load(open(DATA_PATH_BOW + em + '_listing.p', 'r'))
        numb = len(entire_lst)
        indx = random_index(numb - 1, training_count + testing_count)
        sampled_dict[em] = np.array(entire_lst)[indx]

    return sampled_dict



def extract_features_BOW(sampled_dict, training_count, testing_count, tst, normalize = False):
    feature_vector = np.zeros((len(EMOTION_LIST) * training_count, VECTOR_SIZE_BOW))
    test_vector = np.zeros((len(EMOTION_LIST) * testing_count, VECTOR_SIZE_BOW))
    class_vector = []
    test_class = []
    for em in EMOTION_LIST:
        emotion_dict = pickle.load(open(DATA_PATH_BOW + em + '_BOW.p', 'r'))
        vectors = np.zeros((training_count + testing_count, VECTOR_SIZE_BOW))
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

            vectors[count, :] = v
            if count < testing_count:
                test_class += [em]
            else:
                class_vector += [em]
            count += 1

        indx = EMOTION_LIST.index(em)
        offset_tr = training_count * indx
        offset_tst = testing_count * indx
        feature_vector[(0 + offset_tr):(training_count + offset_tr), :] = vectors[0:training_count, :]
        test_vector[(0 + offset_tst):(testing_count + offset_tst), :] = vectors[training_count:, :]

    if normalize:
        print 'Normalizing'
        feature_vector = preprocessing.MinMaxScaler().fit_transform(feature_vector)
        test_vector = preprocessing.MinMaxScaler().fit_transform(test_vector)
    return (feature_vector,
     class_vector,
     test_vector,
     test_class)



def extract_features_Audio(sampled_dict, training_count, testing_count, tst, seg_numb = SEG_NUMB_AUD, normalize = False):
    feature_vector = [None]
    class_vector = []
    test_vector = [None]
    test_class = []
    for em in EMOTION_LIST:
        count = 0
        for fname in sampled_dict[em]:
            f = read_hdf5(DATA_PATH_ExHD + em + '/' + fname + '.h5')
            if not f:
                continue
            seg = hdf.get_segments_timbre(f)
            seg_indx = random_index(seg.shape[0] - 1, seg_numb)
            sampled_seg = seg[seg_indx, :]
            count += 1
            if count > training_count:
                if None in test_vector:
                    test_vector = sampled_seg
                else:
                    test_vector = np.concatenate((test_vector, sampled_seg))
                test_class += [em]
            elif None in feature_vector:
                feature_vector = sampled_seg
            else:
                feature_vector = np.concatenate((feature_vector, sampled_seg))
            class_vector += [em] * seg_numb
            f.close()


    if normalize:
        print 'Normalizing'
        feature_vector = preprocessing.MinMaxScaler().fit_transform(feature_vector)
        test_vector = preprocessing.MinMaxScaler().fit_transform(test_vector)
    return (feature_vector,
     class_vector,
     test_vector,
     test_class)



def post_process_audio(results_arr, seg_numb = SEG_NUMB_AUD):
    size_em = len(EMOTION_LIST)
    total_songs = results_arr.shape[0] / seg_numb
    results = []
    for song in range(total_songs):
        classification = results_arr[(song * seg_numb):(song * seg_numb + seg_numb)]
        class_counter = Counter(classification)
        lst = []
        for em in EMOTION_LIST:
            if em in class_counter:
                lst.append(class_counter[em])
            else:
                lst.append(0)

        results.append(lst)

    return results



def get_Classifier(str):
    clf_dict = {'G': simple_GNB_train,
     'M': simple_MNB_train,
     'S': simple_SVM_train}
    return clf_dict[str]



def simple_SVM_train(feature_vector, class_vector, kernel_type = histogram_intersection_kernel):
    print kernel_type
    clf = svm.SVC(kernel=kernel_type, max_iter=1000, verbose=False, cache_size=1000)
    clf.fit(feature_vector, class_vector)
    return clf



def simple_GNB_train(feature_vector, class_vector, kernel = None):
    clf = GaussianNB()
    clf.fit(feature_vector, class_vector)
    return clf



def simple_MNB_train(feature_vector, class_vector, kernel = None):
    clf = MultinomialNB()
    clf.fit(feature_vector, class_vector)
    return clf



