import db_util



conn = db_util.connect_db()
dat = conn.execute('select description, rating from app_data where not (rating is NULL)')

