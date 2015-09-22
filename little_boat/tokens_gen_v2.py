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
nonsp_file = set(nonsp_file[-len(sp_file):]) # the last instead of the first rows this time

tokens1 = dict()
tokens2 = dict()

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

		for line in f:
			line = line.strip()
			try:
				#token1
				tmp = re.findall('://([A-Za-z0-9][A-Za-z0-9\.]*)',line)
				for t1 in tmp:
					if t1 not in dup_set1:
						if t1 not in tokens1:
							tokens1[t1] = 0
						tokens1[t1] += num
						dup_set1.add(t1)
			except:
				pass
			try:
				#token2
				tmp = re.findall(r"[\S]+",line)
				for t1 in tmp:
					if t1 not in dup_set2:
						if t1 not in tokens2:
							tokens2[t1] = 0
						tokens2[t1] += num
						dup_set2.add(t1)
			except:
				pass
				




# check if ratio is greater than 0.05
threshold1 = int(len(sp_file) * 0.01)
threshold2 = int(len(sp_file) * 0.05)

filtered_token1 = [token for token, count in tokens1.iteritems() if abs(count) > threshold1]
filtered_token2 = [token for token, count in tokens2.iteritems() if abs(count) > threshold2]
print len(filtered_token1),len(filtered_token2)
print filtered_token1[:20]
print filtered_token2[:20]

pickle.dump(filtered_token1, open('token_v2_gain1.p','w'))
pickle.dump(filtered_token2, open('token_v2_gain2.p','w'))





