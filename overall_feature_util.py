import math

def extract_feature_vec(rec) :
    rating = rec[0]
    download_amount = rec[1]
    category = rec[2]
    price = rec[3]
    rating_amount = rec[4]
    app_version = rec[5]
    last_update_date = rec[6]
    sdk_version = rec[7]
    in_app_products = rec[8]
    screenshots_amount = rec[9]
    content_rating = rec[10]

    rating = float(rating)
    download_amount = _extract_download_amount(download_amount)
    category = _extract_category(category)
    price = 0 if price == 'free' else 1
    rating_amount = _extract_rating_amount(rating_amount)
    app_version = _extract_app_version(app_version)
    last_update_date = _extract_last_update_date( last_update_date)
    sdk_version = _extract_sdk_version( sdk_version)
    in_app_products = 0 if in_app_products == None else 1

def _extract_download_amount(x):
    if x == None :
        return 0
    else :
        x = x.replace(',', '')
        x = x.replace('+', '')
        x = int(x)
        return 0 if x == 0 else math.log(x)

_all_game_category = ['GAME_BOARD', 'GAME_TRIVIA', 'FAMILY_BRAINGAMES', 'GAME_ARCADE', 'GAME_CARD', 'GAME_MUSIC', 'GAME_RACING', 
 'GAME_ACTION', 'GAME_PUZZLE', 'GAME_SIMULATION', 'GAME_STRATEGY', 'GAME_ROLE_PLAYING', 'GAME_SPORTS', 'GAME_ADVENTURE', 'GAME_CASINO', 'GAME_WORD', 'GAME_CASUAL', 'GAME_EDUCATIONAL']

def _extract_category(x):
    one_hot_vec = [0] * len(_all_game_category)

    game_category_list = x.split(',')[:-1]
    for game_category in game_category_list :
        try:
            index = _all_game_category.index(game_category)
            if index > -1 :
                one_hot_vec[index] = 1
        except:
            continue
    return one_hot_vec

def _extract_rating_amount(x):
    if x == None :
        return 0
    else :
        x = x.replace(',' , '')
        x = int(x)
        return 0 if x == 0 else math.log(x)

_acceptable_chars = '1234567890.'
def _extract_app_version(x):
    if x == None :
        return 0
    else :
        #filter 
        filtered_acceptable_chars = ''
        for char in x :
            if char in _acceptable_chars :
                filtered_acceptable_chars += char
        
        output = filtered_acceptable_chars.split('.')[0]
        if output == '' : 
            return 0
        else :
            return int(output)

def _extract_last_update_date(x):
    x = x[-4:]
    x = int(x) - 2010
    return x

_all_sdk_version = ['', '4.4 and up', '1.5 and up', '4.1 and up', '2.3 - 7.1.1', '3.2 and up', '2.0.1 and up', '2.1 and up', '3.1 and up', '7.1 and up', '2.3 - 3.2', '8.0 and up', '1.0 and up','6.0 and up', '3.0 and up', '1.6 and up', '4.4W and up', '1.1 and up', '3.0 - 7.0', '2.0 and up', '2.2 and up', '4.0.3 and up', '4.1 - 4.4W', '2.3 - 4.4W', '7.0 and up', '4.0 and up', '4.2 - 7.1.1', '4.0 - 5.0', '4.2 and up', '4.1 - 8.0', '5.0 and up', '4.3 and up', '2.3.3 and up', '2.3 - 7.0', '2.3 and up', '5.1 and up', '2.2 - 4.3', 'Varies with device']
def _extract_sdk_version(x):
    if x == None :
        x = ''
    one_hot_vec = [0] * len(_all_sdk_version)
    index = _all_sdk_version.index(x)
    one_hot_vec[index] = 1
    return one_hot_vec

        
            
