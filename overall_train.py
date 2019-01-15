# import keras
import overall_db_util
from overall_feature_util import extract_feature_vec

#extract feature vector

for record in overall_db_util.query() :
    t = extract_feature_vec(record)
    