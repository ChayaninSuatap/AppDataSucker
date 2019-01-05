import scrapy
import db_util
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

def download_app_data(resp, conn):
    #icon : later 
    #screen shot : later

    #app name
    app_name = resp.css("h1.AHFaub").xpath('.//span/text()').extract_first()

    #description
    description = ''
    desc_lines = resp.css('div.DWPxHb').css('div.DWPxHb').css('content').css('div::text').extract()
    for lines in desc_lines : 
       description += lines + '\n'
    if description == '' :
        description = None

    ### Additional Infomation
    add_info_contents = {}
    download_amount = None

    parent_elems = resp.css('div.hAyfc')
    for container_elem in parent_elems :
        container_name = container_elem.css('div.BgcNfc::text').extract_first()
        if container_name != None :
            container_content = container_elem.css('span.htlgb').css('div.IQ1z0d').css('span.htlgb::text').extract_first()
            if not type(container_content) is list :
                add_info_contents[container_name] = container_content
    
    print(app_name, add_info_contents)
    #extract additional infomation
    for k,v in add_info_contents.items() :
        if k == 'Installs' : download_amount = v

    #save in db
    app_id = get_app_id(resp.url)
    db_util.insert_new_row( app_id , conn)
    db_util.update_game_name(app_name, app_id, conn)
    db_util.update_description(description, app_id, conn)
    if download_amount != None : db_util.update_download_amount( download_amount, app_id, conn)


def get_app_id(app_page_link):
    return app_page_link.split('details?id=')[1]