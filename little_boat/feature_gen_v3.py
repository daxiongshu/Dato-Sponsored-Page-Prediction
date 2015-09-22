'''
generate token feature csv files
'''
import pickle
from csv import DictReader,DictWriter
import glob
import re

# load tokens
token1 = pickle.load(open('token1_gain_v3.p'))
token2 = pickle.load(open('token2_gain_v3.p'))
token3 = pickle.load(open('token3_gain_v3.p'))
token4 = pickle.load(open('token4_gain_v3.p'))

all_tokens = token1 + token2 + token3 + token4
files = glob.glob('data/*/*raw*') # all files

with open("xz_tokens_v3.csv","wb") as outfile:
	fieldnames = list(set(['file']+ all_tokens)) # file and all 4 tokens (and get rid of duplicates)
	writer = DictWriter(outfile, fieldnames = fieldnames)
	writer.writeheader()
	for en,file_path in enumerate(files):
		if en % 1000 == 0:
			print "doing %i files..."%en
		row = dict(zip(all_tokens,[0]*len(all_tokens))) # initialize as zero vector
		row['file'] = file_path.split('/')[-1]
		with open(file_path) as out_f:
			f = out_f.readlines()
			r = int(-0.05 * len(f))
			f = f[r:] # the last 5%
			for line in f:
				line = line.strip()
				#token1
				if line in token1:
					row[line] += 1
				#token2
				try:
					tmp=re.findall(r"[\w']+", line)
					for x in token2:
						row[x] += 1
				except:
					pass
				try:
					#token3
					tmp = re.findall('([A-Za-z0-9][A-Za-z0-9\.]*)',line)
					for x in tmp:
						if x in token3:
							row[x] += 1
				except:
					pass

				#token2
				try:
					tmp = re.findall(r"[\S]+",line)
					for x in tmp:
						if x in token4:
							row[x] += 1
				except:
					pass

		writer.writerow(row)
print "ALL DONE!! The feature csv is ready!"
