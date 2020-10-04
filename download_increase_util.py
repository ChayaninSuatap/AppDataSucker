import db_util
import global_util

def _prepare_dataset(app_ids_d, old_db_path, new_db_path):
    old_conn = db_util.connect_db(old_db_path)
    new_conn = db_util.connect_db(new_db_path)

    old_d = {}
    new_d = {}
    for dat in old_conn.execute('select app_id, download_amount from app_data'):
        if dat[0] in app_ids_d:
            old_d[dat[0]] = int(dat[1].replace(',','').replace('+',''))
    
    for dat in new_conn.execute('select app_id, download_amount from app_data'):
        if dat[0] in app_ids_d:
            new_d[dat[0]] = int(dat[1].replace(',','').replace('+',''))
    
    output = []
    for k_old, v_old in old_d.items():
        if k_old in new_d:
            if new_d[k_old] > v_old:
                output.append((k_old, 1))
            else:
                output.append((k_old, 0))
    return output

def prepare_dataset():
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]:True for x in aial}
    output = _prepare_dataset(
        app_ids_d,
        'crawl_data/first_version/data.db',
        'crawl_data/update_first_version_2020_09_12/data.db')
    return output


if __name__ == '__main__':
    pass


