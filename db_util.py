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

def update_description(desc, app_id, conn):
    conn.execute('UPDATE app_data SET description = ? WHERE app_id = ?', (desc, app_id, ))

def update_download_amount(download_amount, app_id, conn):
    conn.execute('UPDATE app_data SET download_amount = ? WHERE app_id = ?', (download_amount, app_id, ))


if __name__ == '__main__' :
    create_table()


