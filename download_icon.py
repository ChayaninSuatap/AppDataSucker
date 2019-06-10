import requests
import mypath
import db_util
import scrapy
from scrapy_util import get_app_id
from scrapy.crawler import CrawlerProcess
import os.path
from time import sleep

def download_icon(app_id, resp):
    try :
        t = resp.css("div.xSyT2c").css('img.T75of.sHb2Xb').extract()[0]
        t = t.split('"')[1].split('=')[0]
        #get icon file
        with open(mypath.icon_folder + app_id + '.png' , 'wb') as handle:
            t = requests.get(t, stream = True)
            for block in t.iter_content(1024):
                handle.write(block)
        print('downloaded :', app_id)
    except :
        print('error :', app_id)

def _get_downloaded_icon_app_ids():
    output = []
    for fn in os.listdir( mypath.icon_folder) :
        output.append( fn[:-4])
    return output

def _get_undownloaded_icon_app_ids():
    conn = db_util.connect_db()
    app_ids = db_util.get_all_app_id(conn)
    undownloaded = _get_downloaded_icon_app_ids()
    for x in undownloaded :
        app_ids.remove(x)
    return app_ids

class IconDownloader( scrapy.Spider):
    name = 'Joe'
    def start_requests(self) :
        main_link = 'https://play.google.com/store/apps/details?id='
        app_ids = _get_undownloaded_icon_app_ids()

        print('undownloaded amount:', len(app_ids))
        sleep(2)

        for app_id in app_ids :
             yield scrapy.Request( main_link + app_id, self.parse)
    
    def parse(self, resp) :
        app_id = get_app_id(resp.url)
        download_icon(app_id, resp)

if __name__ == '__main__' :
    proc = CrawlerProcess({
    'USER_AGENT': 'Chrome/27.0.1453.93'
    })

    proc.crawl(IconDownloader)
    proc.start()

