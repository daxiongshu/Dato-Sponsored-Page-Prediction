'''
generate token feature csv files
'''
import pickle
from csv import DictReader,DictWriter
import glob
import re

# load tokens
token1 = pickle.load(open('token_v2_gain1.p'))
token2 = pickle.load(open('token_v2_gain2.p'))
all_tokens = token1 + token2
files = glob.glob('data/*/*raw*') # all files

with open("xz_tokens_v2.csv","wb") as outfile:
	fieldnames = list(set(['file']+ all_tokens)) # file and all 4 tokens (and get rid of duplicates)
	writer = DictWriter(outfile, fieldnames = fieldnames)
	writer.writeheader()
	for en,file_path in enumerate(files):
		if en % 1000 == 0:
			print "doing %i files..."%en
		row = dict(zip(all_tokens,[0]*len(all_tokens))) # initialize as zero vector
		row['file'] = file_path.split('/')[-1]
		with open(file_path) as f:
			for line in f:
				line = line.strip()
				try:
					#token1
					tmp = re.findall('://([A-Za-z0-9][A-Za-z0-9\.]*)',line)
					for x in tmp:
						if x in token1:
							row[x] += 1
				except:
					pass

				#token2
			try:
				tmp = re.findall(r"[\S]+",line)
				for x in tmp:
					if x in token2:
						row[x] += 1
			except:
				pass

		writer.writerow(row)
print "ALL DONE!! The feature csv is ready!"
