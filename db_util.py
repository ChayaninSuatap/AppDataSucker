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

def connect_db():
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


