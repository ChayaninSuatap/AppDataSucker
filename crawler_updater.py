import db_util
import scrapy
from scrapy.crawler import CrawlerProcess
import scrapy_util

class CrawlerUpdater( scrapy.Spider):
    name = 'crawler_updater'

    def start_requests(self):
        main_link = 'https://play.google.com/store/apps/details?id='
        self.conn = db_util.connect_db()
        self.app_ids = db_util.get_all_app_id(self.conn)

        for app_id in self.app_ids :
            yield scrapy.Request(main_link + app_id, self.parse)
    
    def parse(self, resp):
        try:
            print('updating :', resp.url)
            scrapy_util.download_app_data(resp, self.conn, update_only = True)
        except Exception as e:
            print(repr(e))
            input()
        
if __name__ == '__main__' :
    proc = CrawlerProcess({
    'USER_AGENT': 'Chrome/27.0.1453.93'
    })

    proc.crawl(CrawlerUpdater)
    proc.start()