from preprocess_util import prep_rating_category_scamount_download
import db_util
import numpy
import matplotlib.pyplot as plt
import os
import global_util

class DBComparer:
    def __init__(self, old_db_path, new_db_path):
        self.old_conn = db_util.connect_db(old_db_path)
        self.new_conn = db_util.connect_db(new_db_path)

    def plot_diff_rating_dist(self):
        old_dat = prep_rating_category_scamount_download(self.old_conn)
        new_dat = prep_rating_category_scamount_download(self.new_conn)

        old_d = {}
        for rec in old_dat:
            old_d[rec[0]] = rec

        new_d = {}
        for rec in new_dat:
            new_d[rec[0]] = rec
        
        diff_ratings = []
        for k,v in new_d.items():
            # filter new app change cate to include "GAME%"
            if k not in old_d:
                continue
            old_rating = old_d[k][1]
            new_rating = v[1]
            diff_ratings.append(new_rating - old_rating)
        plt.hist(diff_ratings, 150)
        plt.show()
    
    def show_diff_download(self, app_ids_d=None):
        old_dat = prep_rating_category_scamount_download(self.old_conn)
        new_dat = prep_rating_category_scamount_download(self.new_conn)

        old_d = {}
        for rec in old_dat:
            if app_ids_d is not None:
                if rec[0] not in app_ids_d:
                    continue
            old_d[rec[0]] = rec

        new_d = {}
        for rec in new_dat:
            if app_ids_d is not None:
                if rec[0] not in app_ids_d:
                    continue
            new_d[rec[0]] = rec
        
        output = {}
        all_key = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000]
        for key1 in all_key:
            output[key1] = {}
            for key2 in all_key:
                output[key1][key2] = 0

        for k,v in new_d.items():
            # filter new app change cate to include "GAME%"
            if k not in old_d:
                continue
            old_download = old_d[k][4]
            new_download = v[4]

            output[old_download][new_download] += 1
        
        for x in output.values():
            for y in x.values():
                print(y, end='\t')
            print()

    def _read_db(self):
        old_dat = prep_rating_category_scamount_download(self.old_conn)
        new_dat = prep_rating_category_scamount_download(self.new_conn)

        old_d = {}
        for rec in old_dat:
            old_d[rec[0]] = rec

        new_d = {}
        for rec in new_dat:
            new_d[rec[0]] = rec
        
        return old_d, new_d
        
    def count_gone_app_ids(self):
        old_d, new_d = self._read_db()
        count = 0
        for k,v in old_d.items():
            if k not in new_d:
                count += 1
        return count
    
    def show_gone_app_ids_download_freq(self):
        old_d, new_d = self._read_db()
        freq = {}
        for k,v in old_d.items():
            if k not in new_d:
                download = v[4]
                if download not in freq:
                    freq[download] = 0
                freq[download] += 1
        sorted_keys = sorted(list(freq.keys()))
        for k in sorted_keys:
            print(k, freq[k])

    def show_reupload_apps(self):
        'warning : ugly code'
        old_d, new_d = self._read_db()
        old_name_d = {}
        new_name_d = {}
        #get old names
        dats = self.old_conn.execute('select app_id ,game_name from app_data')
        for app_id, game_name in dats:
            if app_id in old_d and app_id not in new_d:
                old_name_d[app_id] = game_name
        #get new names
        dats = self.new_conn.execute('select app_id ,game_name from app_data')
        for app_id, game_name in dats:
            if app_id in new_d:
                new_name_d[app_id] = game_name
        #search same game name
        def lcs(s1,s2):
            len1, len2 = len(s1), len(s2)
            ir, jr = 0, -1
            for i1 in range(len1):
                i2 = s2.find(s1[i1])
                while i2 >= 0:
                    j1, j2 = i1, i2
                    while j1 < len1 and j2 < len2 and s2[j2] == s1[j1]:
                        if j1-i1 >= jr-ir:
                            ir, jr = i1, j1
                        j1 += 1; j2 += 1
                    i2 = s2.find(s1[i1], i2+1)
            return s1[ir:jr+1]

        # output = []
        # for old_k in old_d.keys():
        #     if old_k not in new_d:
        #         lcs_best_app_id = None
        #         lcs_best_result = None
        #         lcs_best_n = None
        #         for new_k in new_d.keys():
        #             lcs_result = lcs(old_k, new_k)

        #             if lcs_best_app_id is None:
        #                 lcs_best_app_id = new_k
        #                 lcs_best_result = lcs_result
        #                 lcs_best_n = len(lcs_result)
                
        #         output.append( (old_k, lcs_best_app_id, lcs_best_result, lcs_best_n))
        #         print(old_k, lcs_best_app_id)
        # output = sorted(output, key=lambda x:x[3], reverse=True)
        # f=open('reupload apps.txt','w', encoding='utf-8')
        # for rec in output:
        #     print(rec[0], rec[1], rec[2], rec[3], file=f)
        # f.close()
        

        output = {}
        duplicate_d = {}
        for old_k in old_d.keys():
            if old_k not in new_d:
                for new_k in new_d.keys():
                    # filter game exist in both old and new 
                    if new_k not in old_d and\
                        old_name_d[old_k] == new_name_d[new_k]: # has exactly same name
                        game_name = old_name_d[old_k]

                        if game_name in output:
                            duplicate_d[game_name] = True
                        
                        output[game_name] = (old_k, new_k)
        
        f=open('reupload apps.txt','w', encoding='utf-8')
        sorted_output = []
        for k,v in output.items():
            if k not in duplicate_d:
                lcs_result = lcs(v[0], v[1])
                sorted_output.append([k,v[0], v[1], len(lcs_result)])
        sorted_output = sorted(sorted_output, key=lambda x:x[3], reverse=True)
        for x in sorted_output:
            print(x[0], x[1], x[2], x[3], file=f)
        f.close()

if __name__ == '__main__':
    dbc = DBComparer(
        'crawl_data/first_version/data.db',
        'crawl_data/2020_09_17/data.db'
    )
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]:True for x in aial}
    # dbc.show_gone_app_ids_download_freq()
    dbc.show_diff_download(app_ids_d)



        


