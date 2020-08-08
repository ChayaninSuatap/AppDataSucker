import zipfile
import os
import sys

source_fd = 'c:/screenshots.resized/'
dest_fd = 'c:screenshots.resized.zip/'

def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr   = arr[size:]
    arrs.append(arr)
    return arrs

chrunk_size = 184722 // 6
ss_path = source_fd

fns = os.listdir(ss_path)
chrunks = split(fns, chrunk_size)

for i, chrunk in enumerate(chrunks):
    z = zipfile.ZipFile('%s%d.zip' % (dest_fd, i,), mode='w')
    for fn in chrunk:
        z.write(ss_path + fn, arcname=fn)
        print('added', fn, 'in chrunk', i)
    z.close()

print(len(chrunks[0]) + len(chrunks[1]) + len(chrunks[2]), chrunk_size)


