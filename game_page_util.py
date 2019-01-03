def get_app_category(resp):
    link_to_categories = resp.css('a.hrTbp.R8zArc::attr(href)').extract()[1:]
    game_category_arr = [ x.split('/')[-1] for x in link_to_categories]
    return game_category_arr

def resp_is_game(resp):
    for x in get_app_category(resp):
        if 'GAME' in x : 
            return True
    return False