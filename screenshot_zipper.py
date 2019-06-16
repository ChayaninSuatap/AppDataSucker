import zipfile
import os
import sys

def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr   = arr[size:]
    arrs.append(arr)
    return arrs

chrunk_size = 297009 // 3
ss_path = 'e:/thesis_datasets/screenshots256/'
fns = os.listdir(ss_path)
chrunks = split(fns, chrunk_size)

for i, chrunk in enumerate(chrunks):
    z = zipfile.ZipFile('e:/%d.zip' % (i,), mode='w')
    for fn in chrunk:
        z.write(ss_path + fn, arcname=fn)
    z.close()

print(len(chrunks[0]) + len(chrunks[1]) + len(chrunks[2]), chrunk_size)


