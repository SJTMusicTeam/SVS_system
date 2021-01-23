# 30ms per frame

import numpy as np
import os
import re
import csv

path = '/Users/huonan/Desktop/SVS/Alignment/other_set/' # npy files dir
dest_path = '/Users/huonan/Desktop/SVS/Alignment/other_set/new_data'
phone_path = '/Users/huonan/Desktop/SVS/Alignment/clean_set/new_phone'
phone_file = open(phone_path, 'r', encoding='utf-8')
phone_reader = csv.reader(phone_file, delimiter=' ')
phone_dict = {}
for line in phone_reader:
	phone_dict[int(line[1])] = line[0]

def PackZero(integer):
	pack = 4 - len(str(integer))
	return '0' * pack + str(integer)

other_set = []

files = os.listdir(path)
for file in files:
	name_npy = re.search('.npy', file)
	if name_npy:
		num = file[0:4]
		print (num)
		if num not in other_set:
			other_set.append(num)

			align_name = dest_path+'/'+num+'_align.txt'
			f = open(align_name, 'w')
			f.close()
			start = 1.5

			idx_song = 1
			while True:
				try:
					file = PackZero(idx_song) + '.npy' 

					counter = 1 # count num of frames

					alignment_origin = np.load(path+num+file)
					start += 1.5

					for idx in range(len(alignment_origin)):
						if idx == 0:
							alig_temp = alignment_origin[idx]
							continue
						if alig_temp == alignment_origin[idx]:
							counter += 1
						else:
							if phone_dict[alig_temp] != 'sil': # skip silence
								f = open(align_name, 'a')
								f.write(str(format(start, '.6f')) +'\t' + str(format(start+(0.03*counter), '.6f')) + '\t' + phone_dict[alig_temp] + '\n')
								f.close()
							alig_temp = alignment_origin[idx]
							start = start + (0.03*counter)
							counter = 1
					f = open(align_name, 'a')
					f.write(str(format(start, '.6f')) +'\t' + str(format(start+(0.03*counter), '.6f')) + '\t' + phone_dict[alig_temp] + '\n')
					f.close()
					start += 1.5

					idx_song += 1
				except FileNotFoundError:
					break

