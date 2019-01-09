import requests
import mypath
import db_util
import scrapy
from scrapy_util import get_app_id
from scrapy.crawler import CrawlerProcess

def download_icon(app_id, resp):
    try :
        t = resp.css("div.dQrBL").css('img.T75of.ujDFqe').extract()[0]
        t = t.split('"')[1]
        #get icon file
        with open(mypath.icon_folder + app_id + '.png' , 'wb') as handle:
            t = requests.get(t, stream = True)
            for block in t.iter_content(1024):
                handle.write(block)
        print('downloaded :', app_id)
    except :
        print('error :', app_id)

class IconDownloader( scrapy.Spider):
    name = 'Joe'
    def start_requests(self) :
        main_link = 'https://play.google.com/store/apps/details?id='
        conn = db_util.connect_db()
        app_ids = db_util.get_all_app_id(conn)
        for app_id in app_ids :
             yield scrapy.Request( main_link + app_id, self.parse)
    
    def parse(self, resp) :
        app_id = get_app_id(resp.url)
        download_icon(app_id, resp)
        input()

if __name__ == '__main__' :
    proc = CrawlerProcess({
    'USER_AGENT': 'Chrome/27.0.1453.93'
    })

    proc.crawl(IconDownloader)
    proc.start()
