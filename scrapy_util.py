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

def _download_app_rating_amount(resp):
    try:
        return resp.xpath('''.//span[@class='AYi5wd TBRnV']/span/text()''').extract()[0]
    except:
        return None

def _download_app_screenshots_amount(resp):
    try:
        return len(resp.css('button.NIc6yf').extract())
    except:
        return None

def download_app_data(resp, conn, update_only = False):
    #screen shot : later

    app_name = resp.css("h1.AHFaub").xpath('.//span/text()').extract_first()
    description = _download_app_description(resp)
    category = _download_app_category(resp)
    rating = _download_app_rating(resp)
    price = _download_app_price(resp)
    rating_amount = _download_app_rating_amount(resp)
    screenshots_amount = _download_app_screenshots_amount(resp)
     
    #extract additional infomation
    add_info_contents = _extract_additional_info_data(resp)
    download_amount = None
    app_version = None
    last_update_date = None
    app_size = None
    sdk_version = None
    in_app_products = None
    content_rating = None
        
    for k,v in add_info_contents.items() :
        if k == 'Installs' : download_amount = v
        elif k == 'Current Version' : app_version = v
        elif k == 'Updated' : last_update_date = v
        elif k == 'Size' : app_size = v
        elif k == 'Requires Android' : sdk_version = v
        elif k == 'In-app Products' : in_app_products = v
        elif k == 'Content Rating' : content_rating = v

    print('crawling app :',app_name, download_amount)

    #save in db
    app_id = get_app_id(resp.url)
    if not update_only :
        db_util.insert_new_row( app_id , conn)
    db_util.update_game_name(app_name, app_id, conn)
    db_util.update_description(description, app_id, conn)
    db_util.update_category(category, app_id, conn)
    db_util.update_rating(rating, app_id, conn)
    db_util.update_price(price, app_id,  conn)
    db_util.update_app_version(app_version, app_id, conn)
    db_util.update_last_update_date(last_update_date, app_id, conn)
    db_util.update_app_size(app_size, app_id, conn)
    db_util.update_sdk_version(sdk_version, app_id, conn)
    db_util.update_in_app_products(in_app_products, app_id, conn)
    db_util.update_download_amount( download_amount, app_id, conn)
    db_util.update_rating_amount( rating_amount, app_id, conn)
    db_util.update_screenshots_amount( screenshots_amount, app_id, conn)
    db_util.update_content_rating( content_rating, app_id, conn)

def _download_app_category(resp):
    category = ''
    try:
        link_to_categories = resp.css('a.hrTbp.R8zArc::attr(href)').extract()[1:]
        game_category_arr = [ x.split('/')[-1] for x in link_to_categories]
        for x in game_category_arr:
            category += x + ','
    except:
        category = None
    return category

def _extract_additional_info_data(resp):
    add_info_contents = {}
    parent_elems = resp.css('div.hAyfc')
    for container_elem in parent_elems :
        container_name = container_elem.css('div.BgcNfc::text').extract_first()
        #if got None mean can't find container box
        if container_name != None :
            #got None if not found , got List if multi field
            if container_name == 'Content Rating' :
                content_rating = container_elem.xpath('''.//span/div/span/div/text()''').extract()
                if content_rating != None :
                    output = ''
                    for x in content_rating :
                        output += x + ','
                    add_info_contents['Content Rating'] = output
                else :
                    add_info_contents['Content Rating'] = None
            else :
                container_content = container_elem.css('span.htlgb').css('div.IQ1z0d').css('span.htlgb::text').extract_first()
                #assign None to unextractable container name
                if not type(container_content) is list :
                    add_info_contents[container_name] = container_content
    return add_info_contents

def _download_app_price(resp):
    #free or paid
    try:
        install_button_label = resp.css('button.LkLjZd.ScJHi.HPiPcc.IfEcue::attr(aria-label)').extract()[0]
        if install_button_label=='Install':
            return 'free'
        else:
            return install_button_label
    except:
        return None

def _download_app_description(resp):
    description = ''
    desc_lines = resp.css('div.DWPxHb').css('div.DWPxHb').css('content').css('div::text').extract()
    for lines in desc_lines : 
       description += lines + '\n'
    if description == '' :
        description = None
    return description

def _download_app_rating(resp):
    try:
        return resp.css('div.BHMmbe::text').extract()[0]
    except:
        return None

def get_app_id(app_page_link):
    return app_page_link.split('details?id=')[1]