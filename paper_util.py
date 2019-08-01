from io import BytesIO
import win32clipboard
from PIL import Image
import preprocess_util
import random
import numpy as np

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

if __name__ == '__main__':
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
    random.shuffle(aial)
    aial = [(x[0], np.argmax(x[2])) for x in aial]
    print(aial[:4])
    while True:
        x = random.choice(aial)
        if x[1] == 4:
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



