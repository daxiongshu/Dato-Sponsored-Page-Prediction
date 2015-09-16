import os
import cPickle as pickle
import sys
import re
from datetime import datetime
path='../input/' # where the train test data lives
folder=sys.argv[1]

tokens1={}  # the complete line
tokens2={}  # line.split()[0]
tokens3={}  # line.split()[-1]
tokens4={}  # re.split('<> =:()',line)

if True:
    files=os.listdir(path+folder)
    for d,f in enumerate(files):

        count1={} # the complete line
        count2={} # line.split()[0]
        count3={} # line.split()[-1]
        count4={} # re.split('<> =:()',line)

        fx=open(path+'%s/%s'%(folder,f))
        for c,line in enumerate(fx):
            x=line.strip()
            if x not in count1:
                count1[x]=0
            count1[x]+=1 # count tokens in a single file
 
            try:
                x=line.strip().split()[0]
                if x not in count2:
                    count2[x]=0
                count2[x]+=1
            except:
                pass
            
            try:
                tmp=line.strip().split()
                x=tmp[-1]
                if len(tmp)>1:
                    if x not in count3:
                        count3[x]=0
                    count3[x]+=1
            except:
                pass

            try:
                #tmp=re.split('<> =:()',line.strip()) # this is the same with tokens1! Wrong!
                tmp=re.findall(r"[\w']+", line) # this is what I'm supposed to do

                for x in tmp:
                    if x not in count4:
                        count4[x]=0
                    count4[x]+=1
            except:
                pass

        fx.close()
        for x in count1:
            if count1[x]>c*0.05:
                if x not in tokens1:
                    tokens1[x]=0
                tokens1[x]+=1 # count files

        for x in count2:
            if count2[x]>c*0.05:
                if x not in tokens2:
                    tokens2[x]=0
                tokens2[x]+=1 # count files

        for x in count3:
            if count3[x]>c*0.05:
                if x not in tokens3:
                    tokens3[x]=0
                tokens3[x]+=1 # count files

        for x in count4:
            if count4[x]>c*0.05:
                if x not in tokens4:
                    tokens4[x]=0
                tokens4[x]+=1 # count files

        del count1, count2, count3, count4
        if d%1000==0:
            print '%s folder%s file processed:%d tokens1:%d tokens2:%d tokens3:%d tokens4:%d'%(datetime.now(),folder,d,len(tokens1),len(tokens2),len(tokens3),len(tokens4))


pickle.dump(tokens1,open('tokens1_%s.p'%folder,'w'))
pickle.dump(tokens2,open('tokens2_%s.p'%folder,'w'))
pickle.dump(tokens3,open('tokens3_%s.p'%folder,'w'))
pickle.dump(tokens4,open('tokens4_%s.p'%folder,'w'))


   
