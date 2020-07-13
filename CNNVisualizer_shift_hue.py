from icon_util import load_icon_by_fn, shift_hue
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    app_id = 'com.keesing.android.crossword'
    img_path = 'icons.combine.recrawled/%s.png' % (app_id,)
    img = load_icon_by_fn(img_path, 128, 128)

    for i in range(0,360,20):
        print('hue', i)
        result = shift_hue(img, i/360)
        plt.imshow(result)
        plt.show()

