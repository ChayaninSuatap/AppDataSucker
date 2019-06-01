import scrapy
import db_util
import scrapy_util
import mypath
import requests
from scrapy.crawler import CrawlerProcess

class ScreenshotCrawler( scrapy.Spider):
    name = 'screenshot_crawler'

    def __init__(self):
        self.conn_db = db_util.connect_db()
    
    def start_requests(self):
        main_url = 'https://play.google.com/store/apps/details?id='
        for app_id in db_util.get_all_app_id(self.conn_db):
            link = main_url + app_id
            yield scrapy.Request(link, self.parse)

    def parse(self, resp):
        try:
            app_id = scrapy_util.get_app_id(resp.url)
            scs = resp.xpath('''//*[@class="Q4vdJd"]//img/@data-src''').extract()
            for i,sc_link in enumerate(scs):
                sc_link = sc_link.split('=')[0]
                save_path = mypath.screenshot_folder + app_id + '%2d' % (i,) + '.png'
                
                with open(save_path, 'wb') as handle:
                    t = requests.get(sc_link, stream=True)
                    for block in t.iter_content(1024):
                        handle.write(block)
                
                print('downloaded' ,save_path)
        except Exception as e:
            print(repr(e))
            input()

            
if __name__ == '__main__' :
    proc = CrawlerProcess({
    'USER_AGENT': 'Chrome/27.0.1453.93'
    })

    proc.crawl(ScreenshotCrawler)
    proc.start()
        
