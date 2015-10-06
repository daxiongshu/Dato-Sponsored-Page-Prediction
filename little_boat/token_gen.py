'''
downsample the negative files to same size as positive labels
extract gain tags regardless of frequency, where the abs difference of negative & positive > 0.05
'''
import pickle
from csv import DictReader,DictWriter
import glob
import re

files = glob.glob('data/*/*raw*') # all files
# get the sponsored and non sponsored files
sp_file = list()
nonsp_file = list()
for row in DictReader(open('train_v2.csv')):
	if row['sponsored'] == '0':
		nonsp_file.append(row['file'])
	if row['sponsored'] == '1':
		sp_file.append(row['file'])
sp_file = set(sp_file) # fast look up
nonsp_file = set(nonsp_file[:len(sp_file)])

tokens1 = dict()
tokens2 = dict()
tokens3 = dict()
tokens4 = dict()

for en,file_path in enumerate(files):
	if en % 10000 == 0:
		print "doing %i files..."%en
	with open(file_path) as f:
		ID = file_path.split('/')[-1]
		if ID in sp_file:
			num = 1
		elif ID in nonsp_file:
			num = -1
		else:
			continue
		dup_set1 = set() # every file only count token once
		dup_set2 = set()
		dup_set3 = set()
		dup_set4 = set()
		for line in f:
			line = line.strip()
			#token1
			if line not in dup_set1:
				if line not in tokens1:
					tokens1[line] = 0
				tokens1[line] += num
				dup_set1.add(line)
			#token2
			try:
				x=line.strip().split()[0]
				if x not in dup_set2:
					if x not in tokens2:
						tokens2[x] = 0
					tokens2[x] += num
					dup_set2.add(x)
			except:
				pass
			#token3
			try:
				x=line.strip().split()[-1]
				if x not in dup_set3:
					if x not in tokens3:
						tokens3[x] = 0
					tokens3[x] += num
					dup_set3.add(x)
			except:
				pass
			#token4
			try:
				tmp=re.findall(r"[\w']+", line)
				for x in tmp:
					if x not in dup_set4:
						if x not in tokens4:
							tokens4[x] = 0
						tokens4[x] += num
						dup_set4.add(x)
			except:
				pass





# check if ratio is greater than 0.05
threshold = int(len(sp_file) * 0.01)
filtered_token1 = [token for token, count in tokens1.iteritems() if abs(count) > threshold]
filtered_token2 = [token for token, count in tokens2.iteritems() if abs(count) > threshold]
filtered_token3 = [token for token, count in tokens3.iteritems() if abs(count) > threshold]
filtered_token4 = [token for token, count in tokens4.iteritems() if abs(count) > threshold]
print len(filtered_token1),len(filtered_token2),len(filtered_token3),len(filtered_token4)
pickle.dump(filtered_token1, open('token1_gain.p','w'))
pickle.dump(filtered_token2, open('token2_gain.p','w'))
pickle.dump(filtered_token3, open('token3_gain.p','w'))
pickle.dump(filtered_token4, open('token4_gain.p','w'))




