import scrapy
import scrapy_util
from scrapy.crawler import CrawlerProcess
import link_category_util
import game_page_util
import db_util
from CrawlState import CrawlState

class PlaystoreCrawler( scrapy.Spider):
    name = 'playstore_crawler'

    def __init__(self):
        self.crawl_state = CrawlState()
        self.conn_db = db_util.connect_db()

    def start_requests(self):
        self.crawl_state.add('https://play.google.com/store/apps/category/GAME')
        while self.crawl_state.has_uncrawled_link() :
            for link in self.crawl_state.get_uncrawled_links():
                yield scrapy.Request(link, self.parse)
        
    def parse(self, resp):
        try:
            print(resp.url)
            self.crawl_state.mark_as_crawled(resp.url)
                    
            #link is container
            if link_category_util.link_is_container(resp.url) :
                print('link is container')
                if link_category_util.link_is_cluster(resp.url) :
                    #extract with selenium
                    pass
                else :
                    self.crawl_state.add_links( scrapy_util.extract_all_links(resp))
            #link is app page and is a game
            elif link_category_util.link_is_app_page(resp.url) and game_page_util.resp_is_game(resp) :
                #download app data
                scrapy_util.download_app_data(resp, self.conn_db)
                #and add links
                self.crawl_state.add_links( scrapy_util.extract_all_links(resp))
                
        except Exception as e:
            print(repr(e))
            input()


proc = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})

proc.crawl(PlaystoreCrawler)
proc.start()
