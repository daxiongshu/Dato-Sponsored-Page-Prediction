import cPickle as pickle
import sys
name=sys.argv[1]
tag={}
for i in range(6):
    tmp=pickle.load(open(name+'_%d.p'%i))
    for j in tmp:
        if j not in tag:
            tag[j]=0
        tag[j]+=tmp[j]
c=0
tagx={}
for i in tag:
    if tag[i]==1:
        c+=1
    if tag[i]>10:
        tagx[i]=tag[i]
print c,len(tag),len(tagx)
name=name+'_filt.p'
pickle.dump(tagx,open(name,'w'))
