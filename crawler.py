import scrapy
import scrapy_util
from scrapy.crawler import CrawlerProcess
import link_category_util
import game_page_util
import db_util
import selenium_util
from CrawlState import CrawlState
from time import sleep

class PlaystoreCrawler( scrapy.Spider):
    name = 'playstore_crawler'
    crawl_cluster = True

    def __init__(self, crawl_cluster = True):
        self.crawl_state = CrawlState(save_interval = 10)
        self.conn_db = db_util.connect_db()
        self.selenium_driver = selenium_util.create_selenium_browser()
        self.crawl_cluster = crawl_cluster

    def start_requests(self):
        self.crawl_state.add('https://play.google.com/store/apps/category/GAME')
        
        if self.crawl_cluster :
            while self.crawl_state.has_uncrawled_link() :
                uncrawled_links = self.crawl_state.get_uncrawled_links()
                print('uncrawled links :', len(uncrawled_links))    
                sleep(2)
                for link in uncrawled_links:
                    yield scrapy.Request(link, self.parse)

        #dont crawl cluster
        elif not self.crawl_cluster :
            while self.crawl_state.has_uncrawled_link_no_cluster() :
                uncrawled_links = self.crawl_state.get_uncrawled_links_no_cluster()
                print('uncrawled links :', len(uncrawled_links))    
                sleep(2)
                for link in uncrawled_links:
                    yield scrapy.Request(link, self.parse) 
            
    
    def parse(self, resp):
        try:
            print('crawling :',resp.url)
            #mark as crawled
            self.crawl_state.mark_as_crawled(resp.url)
                    
            #link is container
            if link_category_util.link_is_container(resp.url) :
                if link_category_util.link_is_cluster(resp.url) or link_category_util.link_is_game_category(resp.url):
                        #extract with selenium
                        self.crawl_state.add_links( selenium_util.extract_all_links(resp.url, self.selenium_driver))
                        self.crawl_state.force_save_state()

                else :
                    self.crawl_state.add_links( scrapy_util.extract_all_links(resp))
            #link is app page and is a game
            elif link_category_util.link_is_app_page(resp.url) and game_page_util.resp_is_game(resp) :
                #download app data
                scrapy_util.download_app_data(resp, self.conn_db)
                self.crawl_state.save_state()
                #and add links
                self.crawl_state.add_links( scrapy_util.extract_all_links(resp))
            print('done')                
        except Exception as e:
            print(repr(e))
            input()

if __name__ == '__main__' :
    proc = CrawlerProcess({
    'USER_AGENT': 'Chrome/27.0.1453.93'
    })

    proc.crawl(PlaystoreCrawler)
    proc.start()
