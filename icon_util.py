import numpy as np
from PIL import Image
import mypath

def load_icon_by_app_id(app_id, resizeW, resizeH):
    return open_and_resize(mypath.icon_folder + app_id + '.png', resizeW, resizeH)

def open_and_resize(fn, resizeW, resizeH):
    return np.asarray( _convert_to_rgba(fn, resizeW, resizeH ))[:,:,:3]

def _convert_to_rgba(fn, resizeW, resizeH):
    png = Image.open(fn).convert('RGBA')
    png = png.resize( (resizeW, resizeH))
    background = Image.new('RGBA', png.size, (255,255,255))

    alpha_composite = Image.alpha_composite(background, png)
    return alpha_composite