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

if __name__ == '__main__':
    prep_rating_category()
    

