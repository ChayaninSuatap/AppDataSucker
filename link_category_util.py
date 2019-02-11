def link_is_container(link):
    return link_is_cluster(link) or link_is_dev(link) or link_is_game_category(link)

def link_is_cluster(link):
    return '/collection/cluster' in link

def link_is_dev(link):
    return 'dev?id' in link

def link_is_game_category(link):
    return 'category/GAME' in link

def link_is_app_page(link):
    return 'details?id' in link

def link_is_queueable(link):
    return (link_is_container(link) or link_is_app_page(link)) and link[-1] != '#'
