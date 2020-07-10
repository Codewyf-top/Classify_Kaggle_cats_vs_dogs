import os
from PIL import Image

path = '/home/cyh/视频/Classifer/1.flist'  # the flist path
output = '/home/cyh/视频/Classifer/datasets/12'  # output path
img = []
if not os.path.isdir(output):
    os.mkdir(output)

with open(path, 'r') as f:
    line_f = f.readlines()

for i in range(25000):

    line_f[i] = line_f[i][:-1]
    a1 = line_f[i].split('/')[-1]
    a = a1.split('.')[0]
    # print(a)
    img = Image.open(line_f[i])
    #print(output + '/' + a )
    img.save(output+'/'+ a + '/' +a1)
    print(i)
