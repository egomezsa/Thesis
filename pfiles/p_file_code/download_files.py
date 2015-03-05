import pickle
import sys
import os


ip = '54.173.41.47'
category = 'angry'



def download_file(filepath, category, ip, dest):

	command = 'scp -i ~/Documents/Thesis/security-east.pem ec2-user@'+ip+':'+ filepath + ' ' + dest
	os.system(command)


MAX_COUNT = 2000

song_path = pickle.load(open(category + '.p','r'))


base_path = '/media/msds/data/'
dest =  '/Volumes/My\ Passport\ for\ Mac/FinalDataset/'+category
os.system('mkdir ' + dest)

files = os.listdir('/Volumes/My Passport for Mac/FinalDataset/'+category)
run_count = len(files)


for p in song_path:

	if run_count >= MAX_COUNT:
		break

	abbrv = p[2:5]
	dir_path =  '/'.join(list(abbrv))
	full_path = base_path + dir_path + '/' + p + '.h5'

	if p + '.h5' in files:
		print p
		continue
	run_count += 1
	download_file(full_path,category,ip,dest)







print 'total: ' + str(run_count)

