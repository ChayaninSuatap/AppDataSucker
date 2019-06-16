import icon_util
import os
import scipy.misc

for fn in os.listdir('screenshots'):
    path = 'screenshots/' + fn
    try:
        icon = icon_util.load_icon_by_fn(path, 256, 160, rotate_for_sc=True)
    except:
        continue
    scipy.misc.imsave('screenshots256/' + fn, icon)
    print(fn)