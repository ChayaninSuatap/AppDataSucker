import crawler
import scrapy
from scrapy.crawler import CrawlerProcess

class PlaystoreCrawlerNoCluster(crawler.PlaystoreCrawler):

    def __init__(self):
        super().__init__(crawl_cluster = False)

if __name__ == '__main__' :
    proc = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    proc.crawl(PlaystoreCrawlerNoCluster)
    proc.start()