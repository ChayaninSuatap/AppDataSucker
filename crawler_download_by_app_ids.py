import db_util
import scrapy
from scrapy.crawler import CrawlerProcess
import scrapy_util

class CrawlerDownloadByAppIds( scrapy.Spider):
    name = 'crawler_download_by_app_ids'

    custom_settings={
        'CONCURRENT_REQUESTS':'8'
    }

    def __init__(self, app_ids):
        self.app_ids = app_ids
        self.main_link = 'https://play.google.com/store/apps/details?id='
        self.conn = db_util.connect_db()

    def start_requests(self):
        for app_id in self.app_ids :
            yield scrapy.Request(self.main_link + app_id, self.parse)
    
    def parse(self, resp):
        try:
            print('updating :', resp.url)
            scrapy_util.download_app_data(resp, self.conn)
        except Exception as e:
            print(repr(e))
            input()
        
if __name__ == '__main__' :
    proc = CrawlerProcess({
    'USER_AGENT': 'Chrome/27.0.1453.93'
    })

    conn = db_util.connect_db('crawl_data/first_version/data.db')
    app_ids = db_util.get_all_app_id(conn)

    new_conn = db_util.connect_db()
    new_app_ids = db_util.get_all_app_id(new_conn)
    filtered_app_ids = [x for x in app_ids if x not in new_app_ids]

    proc.crawl(CrawlerDownloadByAppIds, app_ids=filtered_app_ids)
    proc.start()