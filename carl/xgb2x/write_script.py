f=open('run.sh','w')

for i in range(6):
    if i <4:
        f.write('pypy get_token.py %d & \n'%i)
    else:
        f.write('pypy get_token.py %d  \n'%i)


for i in range(1,5):
    f.write('pypy filt.py tokens%d \n'%i)

for i in range(6):
    f.write('pypy build.py %d &\n'%i)

f.close()

