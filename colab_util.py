import os

def get_training_latest_ep(proj, is_icon=True, is_sc=False):
    fns = os.listdir('/content/drive/My Drive/%s' % (proj,))
    ep_nums=[]
    for fn in fns:
        if fn[-3:] == 'png':
            ep_num = int(fn.split('.')[0])
            ep_nums.append(ep_num)
    max_ep = max(ep_nums)

    if is_icon==True:
        return max_ep - (max_ep % 10)
    elif is_sc==True:
        return max_ep - (max_ep % 4)