'''
generate token feature csv files using 16 threads
'''
import pickle,re,glob, sys
from csv import DictReader,DictWriter

thread = int(sys.argv[1])
# load tokens
token1 = pickle.load(open('token1_gain.p'))
token2 = pickle.load(open('token2_gain.p'))
token3 = pickle.load(open('token3_gain.p'))
token4 = pickle.load(open('token4_gain.p'))
all_tokens = token1 + token2 + token3 + token4
files = glob.glob('data/*/*raw*') # all files

with open("xz_tokens_%i.csv"%thread,"wb") as outfile:
	fieldnames = list(set(['file']+ all_tokens)) # file and all 4 tokens (and get rid of duplicates)
	writer = DictWriter(outfile, fieldnames = fieldnames)
	writer.writeheader()
	for en,file_path in enumerate(files):
		if en % 1000 == 0:
			print "doing %i files for %i..."%(en,thread)
		if en % 16 == thread: # take 1/16 out
			row = dict(zip(all_tokens,[0]*len(all_tokens))) # initialize as zero vector
			row['file'] = file_path.split('/')[-1]
			with open(file_path) as f:
				for line in f:
					line = line.strip()
					#token1
					if line in token1:
						row[line] += 1
					#token2
					try:
						x=line.strip().split()[0]
						if x in token2:
							row[x] += 1
					except:
						pass
					#token3
					try:
						x=line.strip().split()[-1]
						if x in tokens3:
							row[x] += 1
					except:
						pass
					#token4
					try:
						tmp=re.findall(r"[\w']+", line)
						for x in tmp:
							if x in tokens4:
								row[x] += 1
					except:
						pass
			writer.writerow(row)
print "The feature csv is ready for thread %i!"%thread
