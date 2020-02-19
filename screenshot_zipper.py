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

chrunk_size = 184722 // 6
ss_path = 'e:/screenshots.distincted.rem.human/'

fns = os.listdir(ss_path)
chrunks = split(fns, chrunk_size)

for i, chrunk in enumerate(chrunks):
    z = zipfile.ZipFile('e:/screenshots.distincted.rem.human.zip/%d.zip' % (i,), mode='w')
    for fn in chrunk:
        z.write(ss_path + fn, arcname=fn)
        print('added', fn, 'in chrunk', i)
    z.close()

print(len(chrunks[0]) + len(chrunks[1]) + len(chrunks[2]), chrunk_size)


