import urllib2 
import time
import json
from pydub import AudioSegment
import numpy as np
import pylab
import pickle
import os
import sys


api_key = 'api_version=3&api_key=d8ce4dd10b053681fca1926e90d14383b621e6f5'
location = '/Volumes/My Passport for Mac/'

def perform_request(request):
	count = 0

	for count in range(0,5):
		try:
			req = urllib2.Request(request)
			resp = urllib2.urlopen(req).read()
			return json.loads(resp)
		except:
			print 'Request error try number' + str(count)



def create_token():
	return perform_request('http://8tracks.com/sets/new.json?'+api_key)

def buildSpectrogram(data):
	filename = 'test.mp3'
	f = open(filename,'w+')
	f.write(data)
	f.close()

	audiofile = AudioSegment.from_file(filename)
	

	frames = np.fromstring(audiofile._data, np.int16)
	Pxx, freqs, t, plot = pylab.specgram(
	    frames,
	    NFFT=4096, 
	    Fs=44100, 
	    detrend=pylab.detrend_none,
	    window=pylab.window_hanning,
	    noverlap=int(4096 * 0.5))
	return Pxx




song_quota = 105
file_quota = 15


tag = sys.argv[1]

mix_req = perform_request('http://8tracks.com/mix_sets/tags:'+tag+'.json?include=mixes&'+api_key)
mixes = mix_req['mix_set']['mixes']
play_token = create_token()
token_id = play_token['play_token']
urlList = ['http://8tracks.com/sets/'+str(token_id)+'/play.json?mix_id=',str(1),'&'+api_key]
reportList = ['http://8tracks.com/sets/111696185/report.json?track_id=','id','&mix_id=','id','&',api_key]


songs_dict = dict()

file_count = 0
file_songs = 0
count = 0
done = False

skips_mixes = int(sys.argv[2])

print len(mixes)
for a in mixes:

	if skips_mixes > 0:
		print a['name']
		skips_mixes = skips_mixes - 1

		continue


	print a['name']
	mix_id = a['id']
	urlList[1] = str(mix_id)
	url = ''.join(urlList)
	resp = perform_request(url)

	while not  resp['set']['at_last_track']:
		track_info =resp['set']['track']
		track_id = track_info['id']

		req = urllib2.Request(track_info['track_file_stream_url'])

		stream = urllib2.urlopen(req)
		data =  stream.read()
		Pxx = buildSpectrogram(data)
		songs_dict[count] = Pxx
		reportList[1]= str(track_id)
		reportList[3]= str(mix_id)
		perform_request(''.join(reportList))
		count = count + 1
		file_songs = file_songs + 1

		if file_songs == file_quota:
			print 'quota'
			f = open(location+tag+str(file_count)+'.p', 'w+')
			pickle.dump(songs_dict,f)
			f.close()
			songs_dict = dict()
			file_count = file_count + 1
			file_songs = 0

		if count == song_quota:
			done = True
			break

		time.sleep(120)
		resp = perform_request('http://8tracks.com/sets/'+str(token_id)+'/next.json?mix_id='+str(mix_id)+'&'+api_key)

		if not resp['set']['skip_allowed']:
			break


		print ('At End: '+ str(resp['set']['at_end']) , 'At last track: ' + str(resp['set']['at_last_track']), 'Skip allowed: ' + str(resp['set']['skip_allowed']))

	if done:
		break


fname = location + tag + str(file_count) + '.p'

f = open(fname, 'w+')
pickle.dump(songs_dict, f)
f.close()


