import scrapy
import link_category_util

def extract_all_links(resp):
    hxs = scrapy.Selector(resp)
    all_links = [_normalize_link(x) for x in hxs.xpath('*//a/@href').extract() if _link_is_within_playstore(x) and link_category_util.link_is_queueable(x)]
    return all_links

def _link_is_within_playstore(link):
    if link[:23] == 'https://play.google.com': return True
    elif link[:6] == 'https:' or link[:5] == 'http:': return False
    else: return True # /app , /dev ..

def _normalize_link(link):
    if not link[:6] == 'https:' :
        return 'https://play.google.com' + link
    else :
        return link

def download_app_data(resp):
    #app name
    app_name = resp.css("h1.AHFaub").xpath('.//span/text()').extract_first()
    #icon : later 
    #screen shot : later
    #description
    description = ''
    desc_lines = resp.css('div.DWPxHb').css('div.DWPxHb').css('content').css('div::text').extract()
    for lines in desc_lines : 
       description += lines + '\n'
    #download amount

    #save in db
    pass