# 30ms per frame

import numpy as np
import os
import re
import csv

path = '/Users/huonan/Desktop/SVS/Alignment/SJT_task/clean/' # label files dir
dest_path = '/Users/huonan/Desktop/SVS/Alignment/SJT_npy/clean'
phone_path = '/Users/huonan/Desktop/SVS/Alignment/other_set/new_phone'
phone_file = open(phone_path, 'r', encoding='utf-8')
phone_reader = csv.reader(phone_file, delimiter=' ')
phone_dict = {}
for line in phone_reader:
	phone_dict[line[0]] = int(line[1])

start_time_prev = -1 
end_time_prev = -1

files = os.listdir(path)
for file_name in files:
	file_txt = file_name[:-4]
	name_npy = re.search('.bak', file_name)
	if not name_npy:
		alignment_new = []
		file = open(path+'/'+file_txt+'.txt','r',encoding='utf-8')
		userlines = file.readlines()
		file.close()
		for line in userlines:
			start_time = float(line.split('\t')[0])
			end_time = float(line.split('\t')[1])
			if (start_time != end_time_prev) and (end_time_prev != -1):
				duration = start_time - end_time_prev
				counter = int(round(duration / 0.03))
				for i in range(counter):
					alignment_new.append(1)

			phone_str = line.split('\t')[2][:-1]

			if ' ' in phone_str:
				phone_str = phone_str.strip()

			duration = end_time - start_time
			counter = int(round(duration / 0.03))
			print(file_name)
			for i in range(counter):
				alignment_new.append(phone_dict[phone_str])

			start_time_prev = start_time 
			end_time_prev = end_time

		np.save(dest_path+'/'+file_txt+'.npy', alignment_new)


