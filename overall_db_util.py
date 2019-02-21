import db_util

def query():
    conn = db_util.connect_db()
    sql = """
    select rating, download_amount, category, price, rating_amount, app_version, last_update_date, sdk_version, in_app_products, screenshots_amount, content_rating,
    video_screenshot from app_data where not app_id like "%&%" and not rating is NULL
    """
    datas = conn.execute(sql)
    for x in datas : 
        yield x
    conn.close()