from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.vanilla_gradients import VanillaGradients
from icon_util import load_icon_by_fn, rgb_to_gray, shift_hue
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
from custom_gradcam import GradCAM as CustomGradCam
from vis.visualization.saliency import visualize_cam, visualize_saliency


cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

def visualize_grad_cam(model, icon, cate_index, save_dest=None, use_custom_gradcam=False,
    show_visualize=True):
    _, ax = plt.subplots(1, 3)
    plot_i = -1
    #get last convolution layer
    last_conv_layer = None
    last_conv_i = None
    for layer_i,layer in enumerate(model.layers):
        if 'conv2d' in layer.name:
            last_conv_layer = layer
            last_conv_i = layer_i
    for layer in model.layers:
        if layer is last_conv_layer:
        # if 'conv2d' in layer.name:
            plot_i += 1
            if not use_custom_gradcam:
                explainer =  GradCAM()
                output = explainer.explain(([icon],None), model, cate_index, layer_name=layer.name, image_weight=0.5)
                explainer =  GradCAM()
                visualize_only = explainer.explain(([icon],None), model, cate_index, layer_name=layer.name, image_weight=0, colormap=cv2.COLORMAP_BONE)
            else:
                # explainer = CustomGradCam(model, cate_index)
                # heatmap = explainer.compute_heatmap(np.array([icon]))
                # icon_new = (icon*255).astype('uint8')
                # _, output = explainer.overlay_heatmap(heatmap, icon_new, colormap=cv2.COLORMAP_VIRIDIS)
                # _, visualize_only = explainer.overlay_heatmap(heatmap, icon_new, colormap=cv2.COLORMAP_BONE, alpha=0)

                visualize_only=visualize_saliency(model, last_conv_i, filter_indices=[cate_index],
                    seed_input=icon)
                output=np.array(visualize_only)

            visualize_only = rgb_to_gray(visualize_only/255)
            magnitude_map = visualize_only[:,:,0]
            applied_magnitude_map = np.zeros_like(icon)
            for i in range(128):
                for j in range(128):
                    applied_magnitude_map[i][j][0] = magnitude_map[i][j] * icon[i][j][0] #red
                    applied_magnitude_map[i][j][1] = magnitude_map[i][j] * icon[i][j][1] #green
                    applied_magnitude_map[i][j][2] = magnitude_map[i][j] * icon[i][j][2] #blue
            sum_magnitude_map = magnitude_map.sum()
            sum_red = applied_magnitude_map[:,:,0].sum()
            sum_green = applied_magnitude_map[:,:,1].sum()
            sum_blue = applied_magnitude_map[:,:,2].sum()
            print(sum_red, sum_green, sum_blue, sum_magnitude_map)
            print(sum_red/sum_magnitude_map, sum_green/sum_magnitude_map, sum_blue/sum_magnitude_map)

            ax[0].imshow(icon)
            ax[1].imshow(output)
            ax[2].imshow(visualize_only)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
    if save_dest is not None:
        plt.savefig(save_dest)
    elif show_visualize:
        plt.show()
    return np.array([sum_red/sum_magnitude_map, sum_green/sum_magnitude_map, sum_blue/sum_magnitude_map])

def visualize_with_explain_fn(model, icon, cate_index, explain_fn, save_dest=None):
    _, ax = plt.subplots(1, 2)
    plot_i = -1
    #get last convolution layer
    last_conv_layer = None
    for layer in model.layers:
        if 'conv2d' in layer.name:
            last_conv_layer = layer
    for layer in model.layers:
        if layer is last_conv_layer:
        # if 'conv2d' in layer.name:
            plot_i += 1
            # output = explainer.explain(([icon],None), model, cate_index, patch_size=4)
            output = explain_fn(icon, model, cate_index)
            ax[0].imshow(icon)
            ax[1].imshow(output)
            ax[0].axis('off')
            ax[1].axis('off')
    if save_dest is not None:
        plt.savefig(save_dest)
    else:
        plt.show()    

def make_ocs_explain_fn(patch_size):
    def fn(icon, model, cate_index):
        exp = OcclusionSensitivity()
        return exp.explain(([icon], None), model, cate_index, patch_size)
    return fn

def make_vanilla_grad_explain_fn():
    def fn(icon, model, cate_index):
        exp = VanillaGradients()
        return exp.explain((np.array([icon]), None), model, cate_index)
    return fn

if __name__ == '__main__':
    app_id = 'com.keesing.android.crossword'
    cate_index = 3
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    img_path = 'icons.combine.recrawled/%s.png' % (app_id,)
    model = load_model(model_path)

    for i in range(0, 360, 20):
        img = load_icon_by_fn(img_path, 128, 128)
        icon = shift_hue(img, i/360)/255

        pred = model.predict(np.array([icon]))
        conf = max(pred[0])
        cate_index = np.argmax(pred[0])
        print('pred cate', cates[cate_index], 'max pred', pred[0].max())

    plt.imshow(img)
    plt.show()