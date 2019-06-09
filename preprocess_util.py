import db_util
from overall_feature_util import _extract_category
def prep_rating_category():
    conn = db_util.connect_db()
    app_ids_and_labels = []
    dat=conn.execute('select app_id, rating, category from app_data')
    for x in dat:
        if x[1] != None and x[2] != None:
            app_ids_and_labels.append( (x[0], x[1], x[2])) 

    #rating str to float
    for i in range(len(app_ids_and_labels)):
        app_id , rating, cate = app_ids_and_labels[i]
        rating = float(app_ids_and_labels[i][1])
        app_ids_and_labels[i] = app_id, rating, cate
    
    #prep category
    for i,x in enumerate(app_ids_and_labels):
        onehot = _extract_category(x[2])
        app_ids_and_labels[i] = x[0], x[1], onehot
    
    return app_ids_and_labels

def prep_rating_category_scamount_download(for_softmax=False):
    conn = db_util.connect_db()
    app_ids_and_labels = []
    dat=conn.execute('select app_id, rating, category, screenshots_amount, download_amount, rating_amount  from app_data')
    for x in dat:
        if x[1] != None and x[2] != None and x[3] != None and x[4] != None and x[5] != None:
            app_ids_and_labels.append( (x[0], x[1], x[2], x[3], x[4], x[5])) 

    #rating str to float
    output = []
    for i in range(len(app_ids_and_labels)):
        app_id , rating, cate, scamount, download, rating_amount = app_ids_and_labels[i]
        rating = float(rating)
        scamount = float(scamount)
        #download 
        download = download.replace(',','').replace('+','')
        download = int(download)
        #rating amount
        rating_amount = rating_amount.replace(',','')
        rating_amount = int(rating_amount)
        #category
        cate = _extract_category(cate)
        if for_softmax:
            cate[2] = 0
        if all(y==0 for y in cate):
            continue
        output.append((app_id, rating, cate, scamount, download, rating_amount))
    
    return output

def remove_low_rating_amount(aial, threshold):
    newaial = []
    for x in aial:
        if x[5] > threshold:
            newaial.append(x)
    return newaial


if __name__ == '__main__':
    prep_rating_category()
    

