import pickle
import os
import sys

def count_word(word):

	song_list = []

	for file in os.listdir('.'):
		if 'sorted' in file:
			
			word_list = pickle.load(open(file,'r'))

			if word in word_list:
				song_list = song_list + word_list[word]

	return song_list

def export_word(word):

	word_set =  list(set(count_word(word)))

	dmp = open(word+'.p', 'wb')
	pickle.dump(word_set,dmp)
	dmp.close()

	return len(word_set)



words = ['melancholic', 'cheerful', 'good', 'love', 'wrath', 'riot', 'angst', 'emo', 'scream', 'sad', 'dark', 'angsty', 'rebel', 'fun', 'mellow', 'angry', 'hate', 'joyful', 'awesome', 'happy']

runsum = 0
f = open('summary.txt', 'wb')
for w in words:
	print w
	count = export_word(w)
	f.write(w + ': ' + str(count)+'\n')
	runsum += count

f.write('Total: ' + str(runsum))
f.close()