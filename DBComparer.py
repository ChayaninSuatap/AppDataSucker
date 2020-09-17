from preprocess_util import prep_rating_category_scamount_download
import db_util
import numpy
import matplotlib.pyplot as plt

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
    
    def show_diff_download(self):
        old_dat = prep_rating_category_scamount_download(self.old_conn)
        new_dat = prep_rating_category_scamount_download(self.new_conn)

        old_d = {}
        for rec in old_dat:
            old_d[rec[0]] = rec

        new_d = {}
        for rec in new_dat:
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

if __name__ == '__main__':
    dbc = DBComparer(
        'crawl_data/first_version/data.db',
        'crawl_data/2020_09_12/data.db'
    )
    dbc.show_gone_app_ids_download_freq()



        


