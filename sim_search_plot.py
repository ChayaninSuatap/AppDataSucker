from global_util import load_pickle, save_pickle
import os
import icon_util
from similarity_search_util import filter_non_original_sc_human_set_from_sc_hpc, get_human_testset_cate_from_fn, make_category_dict
from sim_search_icon_and_sc import check_sim_search_accuracy
from PIL import Image
import matplotlib.pyplot as plt
import shutil

# filter_non_original_sc_human_set_from_sc_hpc('sim_search_t/sc_hpc/s9_f_e.obj', 'sim_search_t/sc_hpc/s9_f_e_only_origin.obj')

# input(load_pickle('sim_search_t/topn_cache/i13_s13_p_top10.obj'))


def save_in_plot_cache(source_path, fn_index, plot_cache_path = 'sim_search_t/plot_cache/'):
    shutil.copyfile(source_path, plot_cache_path + str(fn_index) + '.png')

def plot_icon_and_sc_topn10(main_fn, topn_list, human_app_ids_path = 'app_ids_for_human_test.obj'):
    cate_dict = make_category_dict()
    icon_fd = 'icons.combine.recrawled/'
    sc_fd = 'screenshots/'

    _, ax = plt.subplots(2, 6, figsize=(8,4))    
    [x.set_axis_off() for x in ax.ravel()]

    #get app_id
    main_fn_index = int(main_fn[:-4])
    app_id = load_pickle(human_app_ids_path)[main_fn_index][0]
    print('app_id',app_id)
    
    ax[0,0].imshow(Image.open(icon_fd + app_id + '.png'))
    ax[0,0].text(0, 0.5, cate_dict[app_id])
    save_in_plot_cache(icon_fd + app_id + '.png', 0)

    #show sc of main app id
    for i in range(1, 5):
        sc_path = sc_fd + app_id + ' ' + str(i-1) + '.png'
        ax[0,i].imshow(Image.open(sc_path))
        save_in_plot_cache(sc_path, i)

    #show top5
    for i in range(5):
        icon_path = icon_fd + topn_list[i][0]
        sc_path = sc_fd + topn_list[i][0]

        try:
            ax[1,i].imshow(Image.open(icon_path))
            ax[1,i].text(0, 0.5, cate_dict[topn_list[i][0][:-4]])
            save_in_plot_cache(icon_path, i+10)
        except:
            ax[1,i].imshow(Image.open(sc_path))
            ax[1,i].text(0, 0.5, cate_dict[topn_list[i][0][:-6]])
            save_in_plot_cache(sc_path, i+10)

    plt.show()

    

def plot_human_icon_top10():
    cate_dict = make_category_dict()
    
    # human_fd = 'icons_human_test/'
    human_fd = 'screenshots_human_test/'
    # img_fd = 'icons.combine.recrawled/'
    img_fd = 'screenshots/'

    _, ax = plt.subplots(2, 6, figsize=(8,4))    
    [x.set_axis_off() for x in ax.ravel()]

    ax[0,0].imshow(Image.open(human_fd + min_main_fn))
    ax[0,0].text(0, 0.5, get_human_testset_cate_from_fn(min_main_fn))
    ax[1,0].imshow(Image.open(human_fd + max_main_fn))
    ax[1,0].text(0, 0.5, get_human_testset_cate_from_fn(max_main_fn))
    print(min_main_fn, max_main_fn)

    save_in_plot_cache(human_fd + min_main_fn, 0)
    for i in range(5):
        save_in_plot_cache(img_fd + min_topn[i][0], i + 1)


    for i in range(1,6):
        ax[0, i].imshow(Image.open(img_fd + min_topn[i-1][0]))
        ax[0, i].text(0, 0.5, cate_dict[min_topn[i-1][0][:-6]] + ' ' + str(min_topn[i-1][1]))
        ax[1, i].imshow(Image.open(img_fd + max_topn[i-1][0]))
        ax[1, i].text(0, 0.5, cate_dict[max_topn[i-1][0][:-6]] + ' ' + str(min_topn[i-1][1]))

    # for i in range(5):
    #     current_key = random.choice(list(output.keys())) #sample a human testset icon fn
    #     ax[i,0].imshow(Image.open(human_icon_fd + current_key))
    #     ax[i,0].text(0, 0.5, get_human_testset_cate_from_fn(current_key))
    #     for j in range(1,6):
    #         ax[i,j].imshow(Image.open(icons_fd + output[current_key][j-1]))
    #         print(cate_dict[output[current_key][j-1][:-4]])
    #         print(output[current_key][j-1][:-4])
    #         ax[i,j].text(0, 0.5, '%s \n%s' % (cate_dict[output[current_key][j-1][:-4]], output[current_key][j-1][:-4]))
    plt.show()

if __name__ == '__main__':
    min_main_fn, min_topn, max_main_fn, max_topn = check_sim_search_accuracy('sim_search_t/topn_cache/i13_s13_p_top10.obj')
    # print(min_main_fn)
    # print(min_topn)
    print(min_main_fn)
    print(min_topn)
    plot_icon_and_sc_topn10(min_main_fn, min_topn)
    # plot_human_icon_top10()