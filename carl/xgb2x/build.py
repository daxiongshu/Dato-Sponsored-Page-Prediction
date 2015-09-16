import re
import os
import cPickle as pickle
import csv
import sys
from datetime import datetime
folder=sys.argv[1]
path='../input/'

tokens1=pickle.load(open('tokens1_filt.p'))
tokens2=pickle.load(open('tokens2_filt.p'))
tokens3=pickle.load(open('tokens3_filt.p'))
tokens4=pickle.load(open('tokens4_filt.p'))


train={} # id -> label
for row in csv.DictReader(open(path+'train_v2.csv')):
    train[row['file']]=row['sponsored']
print len(train)
 


if True:
    files=os.listdir(path+folder)
    for d,f in enumerate(files):
        
        count1={} # complete line
        count2={} # line.split()[0]
        count3={} # line.split()[-1]
        count4={} # re.split('<> =:()',line)

        for j in tokens1:
            count1[j]=0
        for j in tokens2:
            count2[j]=0
        for j in tokens3:
            count3[j]=0
        for j in tokens4:
            count4[j]=0
        fx=open(path+'%s/%s'%(folder,f))
        for c,line in enumerate(fx):
            x=line.strip()
            if x in count1:
                count1[x]+=1
 
            try:
                x=line.strip().split()[0]
                if x in count2:
                    count2[x]+=1
            except:
                pass
            
            try:
                tmp=line.strip().split()
                x=tmp[-1]
                if len(tmp)>1:
                    if x  in count3:
                        count3[x]+=1
            except:
                pass

            try:
                #tmp=re.split('<> =:()',line.strip()) # this is the same with tokens1! Wrong!
                tmp=re.findall(r"[\w']+", line) # this is what I'm supposed to do

                for x in tmp:
                    if x in count4:
                        count4[x]+=1
            except:
                pass
                 
        if f in train:
            if os.path.exists('train%s.csv'%folder):
                fo=open('train%s.csv'%folder,'a')
                cc=','.join(['%d'%count1[j] for j in tokens1]+['%d'%count2[j] for j in tokens2]+['%d'%count3[j] for j in tokens3]+['%d'%count4[j] for j in tokens4])
                line='%s,%s,%s\n'%(f,train[f],cc)
                fo.write(line)
                fo.close()
            else:
                fo=open('train%s.csv'%folder,'w')
                cc=','.join(['fea%d'%k for k in range(len(tokens1)+len(tokens2)+len(tokens3)+len(tokens4))])
                fo.write('file,sponsored,%s\n'%cc)
                cc=','.join(['%d'%count1[j] for j in tokens1]+['%d'%count2[j] for j in tokens2]+['%d'%count3[j] for j in tokens3]+['%d'%count4[j] for j in tokens4])
                line='%s,%s,%s\n'%(f,train[f],cc)
                fo.write(line)
                fo.close()
        elif folder=='5':
            if os.path.exists('test.csv'):
                fo=open('test.csv','a')
                cc=','.join(['%d'%count1[j] for j in tokens1]+['%d'%count2[j] for j in tokens2]+['%d'%count3[j] for j in tokens3]+['%d'%count4[j] for j in tokens4])
                line='%s,%s\n'%(f,cc)
                fo.write(line)
                fo.close()
            else:
                fo=open('test.csv','w')
                cc=','.join(['fea%d'%k for k in range(len(tokens1)+len(tokens2)+len(tokens3)+len(tokens4))])               
                fo.write('file,%s\n'%cc)
                cc=','.join(['%d'%count1[j] for j in tokens1]+['%d'%count2[j] for j in tokens2]+['%d'%count3[j] for j in tokens3]+['%d'%count4[j] for j in tokens4])
                line='%s,%s\n'%(f,cc)
                fo.write(line)
                fo.close()
        if d%1000==0:       
             print '%s folder%s file processed:%d'%(datetime.now(),folder,d)
 
        del count1,count2,count3,count4

