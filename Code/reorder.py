import pickle


DATA_PATH = '../pfiles/'
em_lst = ['sad', 'happy', 'angry']


for em in em_lst:

    pdist = pickle.load(open(DATA_PATH + em + '_BOW.p'))
    file_lst = []

    for m in pdist:
        file_lst.append(m)

    file_lst.sort()

    pickle.dump(file_lst, open(DATA_PATH + em + '_listing.p','w'))

