import os

def get_training_latest_ep(proj, is_icon=False, is_sc=False, custom_period=None):
    fns = os.listdir('/content/drive/My Drive/%s' % (proj,))
    ep_nums=[]
    for fn in fns:
        if fn[-3:] == 'png':
            ep_num = int(fn.split('.')[0])
            ep_nums.append(ep_num)

    if len(ep_nums) == 0:
        return 0
    else:
        max_ep = max(ep_nums)

    if custom_period is not None:
        return max_ep - (max_ep % custom_period)
    elif is_icon==True:
        return max_ep - (max_ep % 10)
    elif is_sc==True:
        return max_ep - (max_ep % 4)