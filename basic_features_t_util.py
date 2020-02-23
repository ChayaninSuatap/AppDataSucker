import os
from preprocess_util import filter_sc_fns_from_icon_fns_by_app_id

def make_not_computed_gist_sc_list(old_sc_gist_txt, sc_fns):
    old_computed_list = {}
    for line in open(old_sc_gist_txt):
        splited = line.split(' ')[:-512]

        if len(splited) == 1:
            sc_fn = splited[0]
        elif len(splited) == 2:
            sc_fn = splited[0] + splited[1]
        else:
            raise Exception('splited > 2')
        
        old_computed_list[sc_fn] = 1
    
    not_computed_sc_list = []
    for sc_fn in sc_fns:
        if sc_fn not in old_computed_list:
            not_computed_sc_list.append(sc_fn)
    
    print(not_computed_sc_list)
    print(len(not_computed_sc_list))
    


if __name__ == '__main__':
    sc_fns = filter_sc_fns_from_icon_fns_by_app_id('similarity_search/icons_rem_dup_human_recrawl/', 'screenshots.256.distincted.rem.human/')
    make_not_computed_gist_sc_list('basic_features/gistdescriptor/gist.sc.txt', sc_fns)