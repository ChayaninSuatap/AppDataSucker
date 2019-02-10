import math

def extract_feature_vec(rec , use_download_amount = True, use_rating_amount = True, use_sdk_version=True, use_last_update_date=True, use_screenshots_amount=True, use_price=True \
,use_content_rating=True, use_app_version=True, use_category=True, use_in_app_products=True):
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

    if float(rating) <= 3.5: rating = 0
    elif float(rating) > 3.5 and float(rating) <= 4.0: rating = 1
    elif float(rating) > 4.0 and float(rating) <= 4.5: rating = 2
    else: rating = 3
    download_amount = _extract_download_amount(download_amount)
    category = _extract_category(category)
    price = 0 if price == 'free' else 1
    rating_amount = _extract_rating_amount(rating_amount)
    app_version = _extract_app_version(app_version)
    last_update_date = _extract_last_update_date( last_update_date)
    sdk_version = _extract_sdk_version( sdk_version)
    in_app_products = 0 if in_app_products == None else 1
    screenshots_amount = _extract_screenshots_amount( screenshots_amount)
    content_rating = _extract_content_rating( content_rating)

    output_vec = []
    if use_in_app_products : output_vec += [in_app_products]
    if use_download_amount : output_vec += [download_amount]
    if use_rating_amount : output_vec += [rating_amount]
    if use_sdk_version : output_vec += sdk_version
    if use_last_update_date : output_vec += [last_update_date]
    if use_screenshots_amount : output_vec += [screenshots_amount]
    if use_price : output_vec += [price]
    if use_content_rating : output_vec += content_rating
    if use_app_version : output_vec += [app_version]
    if use_category : output_vec += category
    return output_vec , rating


_all_content_rating = \
['Gambling', 'Rated for 12+', ' Mild Violence', ' Strong Language', 'Strong Violence', 'Drugs', ' Sexual Innuendo', ' Simulated Gambling', ' Sex', 'Mild Swearing', ' Nudity',
'Rated for 16+', ' Fear', 'Horror', ' Drugs', 'Strong Language', ' Horror', 'Moderate Violence', ' Moderate Violence', 'Fear', ' Implied Violence', 'Nudity', 'Use of Alcohol/Tobacco', ' Gambling', 'Mild Violence', 'Rated for 3+', 'Parental Guidance Recommended', 'Sexual Innuendo', 'Unrated', 'Implied Violence', 'Rated for 7+', 'Simulated Gambling', 'Rated for 18+', ' Mild Swearing', 'Sex', 'Extreme Violence', ' Use of Alcohol/Tobacco', 'Warning – content has not yet been rated.']
def _extract_content_rating(x):
    one_hot_vec = [0] * len( _all_content_rating)
    splited = x.split(',')[:-1]
    for content_rating in splited :
        try :
            index = _all_content_rating.index(content_rating)
            one_hot_vec[index] = 1
        except :
            pass
    return one_hot_vec

def _extract_screenshots_amount(x):
    if x == None :
        return 0
    else :
        return int(x)

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

        
            
