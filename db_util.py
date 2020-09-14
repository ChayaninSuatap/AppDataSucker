import sqlite3
from datetime import date
import mypath

def create_table():
    create_table_sql = """
    CREATE TABLE app_data (
        app_id text primary key,
        game_name text,
        download_amount text,
        description text,
        category text,
        crawl_date date
    );
    """
    try:
        conn = sqlite3.connect(mypath.data_db)
        conn.execute(create_table_sql)
        return True
    except Exception as e:
        print(repr(e))
        return False

def connect_db(db_path=None):
    if db_path is not None:
        return sqlite3.connect(db_path)
    else:
        return sqlite3.connect(mypath.data_db)

def insert_new_row(app_id, conn):
    try:
        conn.execute('INSERT INTO app_data (app_id, crawl_date) VALUES (?, ?)', (app_id, date.today()))
        conn.commit()
    except:
        #by pass insert same app_id
        print('duplicate app_id :', app_id)
        pass

def update_game_name(game_name, app_id, conn):
    conn.execute('UPDATE app_data SET game_name = ? WHERE app_id = ?', (game_name, app_id, ))
    conn.commit()

def update_description(desc, app_id, conn):
    conn.execute('UPDATE app_data SET description = ? WHERE app_id = ?', (desc, app_id, ))
    conn.commit()

def update_download_amount(download_amount, app_id, conn):
    conn.execute('UPDATE app_data SET download_amount = ? WHERE app_id = ?', (download_amount, app_id, ))
    conn.commit()

def update_category(category , app_id, conn):
    _update_field('category', category, app_id, conn)

def update_rating(rating, app_id, conn):
    _update_field('rating', rating, app_id, conn)

def update_price(price, app_id, conn):
    _update_field('price', price, app_id, conn)

def update_rating_amount(rating_amount, app_id, conn):
    _update_field('rating_amount', rating_amount, app_id, conn)

def update_video_screenshot(video_screenshot, app_id, conn):
    _update_field('video_screenshot', video_screenshot, app_id, conn)

def update_last_update_date(value, app_id, conn):
    _update_field('last_update_date', value, app_id, conn)

def update_app_size(value, app_id, conn):
    _update_field('app_size', value, app_id, conn)

def update_content_rating(value, app_id, conn):
    _update_field('content_rating', value, app_id, conn)

def update_screenshots_amount(value, app_id, conn):
    _update_field('screenshots_amount', value, app_id, conn)

def update_sdk_version(value, app_id, conn):
    _update_field('sdk_version', value, app_id, conn)

def update_in_app_products(value, app_id, conn):
    _update_field('in_app_products', value, app_id, conn)

def update_app_version(app_version, app_id, conn):
    _update_field('app_version', app_version, app_id, conn)

def _update_field(field_name , field_value, app_id, conn):
    conn.execute('UPDATE app_data SET ' + field_name + '= ? WHERE app_id = ?', (field_value, app_id, ))
    conn.commit()    

def get_all_app_id(conn):
    datas = conn.execute('SELECT app_id from app_data')
    output = []
    for row in datas :
        output.append( row[0])
    return output


if __name__ == '__main__' :
    # create_table()
    print(get_all_app_id(connect_db()))


