import pickle
import os
import sys


def export_list(arg):
	new_l = []

	for k in arg:
		new_l += arg[k]

	return new_l

file_list = []

for file in os.listdir('.'):
	if 'sorted' in file:
		file_list.append(file)

print file_list
total = 0
intersection = 0

for f in range(len(file_list) - 1):
	l1 = export_list(pickle.load(open(file_list[f],'r')))
	l2 = export_list(pickle.load(open(file_list[f+1], 'r')))

	s1 = set(l1)
	s2 = set(l2)

	total += len(s1)
	intersection += len(s1) + len(s2)  -  len( set(l1 + l2))

total += len(s2)



print (total, intersection, total-intersection)

