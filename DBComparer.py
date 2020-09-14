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



if __name__ == '__main__':
    db_comparer = DBComparer(
        'crawl_data/first_version/data.db',
        'crawl_data/2020_09_12/data.db'
    )
    db_comparer.plot_diff_rating_dist()
        


