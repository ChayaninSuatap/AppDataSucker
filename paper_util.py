from io import BytesIO
import win32clipboard
from PIL import Image
import preprocess_util
import random
import numpy as np
import sc_util
import PIL

def copy_to_clipboard(image):
    def _send_to_clipboard(clip_type, data):
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(clip_type, data)
        win32clipboard.CloseClipboard()

    output = BytesIO()
    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()

    _send_to_clipboard(win32clipboard.CF_DIB, data)

def copy_random_icon_to_clipboard(category_index):
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
    random.shuffle(aial)
    aial = [(x[0], np.argmax(x[2])) for x in aial]
    print(aial[:4])
    while True:
        x = random.choice(aial)
        if x[1] == category_index:
            try:
                png = Image.open('icons.512/%s.png' % x[0]).convert('RGBA')
                background = Image.new('RGBA', png.size, (255,255,255))
                alpha_composite = Image.alpha_composite(background, png)
                copy_to_clipboard(alpha_composite)
                print('copied', x[0])
                input()
            except Exception as e:
                print(e)
                # input()
                pass

def copy_random_sc_to_clipboard(category_index, use_resize, resizew=None, resizeh=None):
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
    random.shuffle(aial)
    aial = [(x[0], np.argmax(x[2])) for x in aial]
    sc_dict = sc_util.make_sc_dict()

    category_dict = {}
    for x in aial:
        category_dict[x[0]] = x[1]

    while True:
        app_id = random.choice(list(sc_dict.keys()))
        sc_fn = random.choice(list(sc_dict[app_id]))
        if app_id in category_dict and category_dict[app_id] == category_index:
            try:
                png = Image.open('screenshots/%s' % sc_fn).convert('RGBA')
                w,h = png.size
                if w > h : png = png.rotate(-90, expand=True)
                if use_resize:
                    png = png.resize( (resizew, resizeh))
                copy_to_clipboard(png)
                print('copied', sc_fn)
                input()

            except Exception as e:
                print(e)
                # input()
                pass

if __name__ == '__main__':
   copy_random_sc_to_clipboard(4, use_resize=False)

'''
BOARD	TRIVIA	ARCADE	CARD	MUSIC	RACING	ACTION	PUZZLE	SIMULATION	STRATEGY	ROLE_PLAYING	SPORTS	ADVENTURE	CASINO	WORD	CASUAL	EDUCATIONAL
0	    1	    2	    3   	4	    5   	6	    7   	8	        9	        10              	11	12	        13  	14	    15  	16
'''

