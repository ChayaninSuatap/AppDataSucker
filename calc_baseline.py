import db_util

conn = db_util.connect_db()
data = conn.execute('SELECT rating FROM app_data where (not rating is NULL) and (not app_id LIKE "%&%")')

ratings = [float(x[0]) for x in data]
print(sum(ratings)/len(ratings))
