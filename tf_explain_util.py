from tf_explain.core.grad_cam import GradCAM
from icon_util import load_icon_by_fn
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

def visualize_grad_cam(model, icon, cate_index):
    _, ax = plt.subplots(5, 3)
    plot_i = -1
    for layer in model.layers:
        if 'conv2d' in layer.name:
            plot_i += 1
            explainer =  GradCAM()
            output = explainer.explain(([icon],None), model, cate_index, layer_name=layer.name, image_weight=0.5)
            explainer =  GradCAM()
            visualize_only = explainer.explain(([icon],None), model, cate_index, layer_name=layer.name, image_weight=0,)
            ax[plot_i, 0].imshow(icon)
            ax[plot_i, 1].imshow(output)
            ax[plot_i, 2].imshow(visualize_only)
    plt.show()

if __name__ == '__main__':
    app_id = 'com.radefffactory.cardsbattle'
    cate_index = 3
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    img_path = 'icons.combine.recrawled/%s.png' % (app_id,)
    img = load_icon_by_fn(img_path, 128, 128)/255
    icon = np.array(img)
    model = load_model(model_path)
    visualize_grad_cam(model, icon, cate_index)
