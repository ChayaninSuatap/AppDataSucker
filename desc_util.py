from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np

def indexize_words(xs, num_words = 2500):
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(xs)
    sequences = tokenizer.texts_to_sequences(xs)
    return pad_sequences(sequences)

def preprocess_text(descs, labels, non_english_char_threshold=0.7):
    english_only_descs = []
    output_labels = []
    threshold = non_english_char_threshold
    for x,y in zip(descs, labels):
        count_bad = 0
        count_good = 0
        english_only_words = []
        for c in x:
            try:
                t = c.encode('ascii')
                count_good+=1
                english_only_words.append(c)
            except:
                count_bad+=1
        ratio = count_bad/(count_bad+count_good)
        if ratio < threshold:
            english_only_descs.append( ''.join(english_only_words))
            output_labels.append(y)
    descs = [x.lower() for x in english_only_descs]
    descs = [re.sub('[^a-zA-z0-9\s]','',x) for x in descs]
    return descs, np.array(output_labels)

# dist.append((app_id, count_good, count_bad, count_bad/(count_bad+count_good)))
# count_bad = 0
# count_good = 0
# for x in dist:
#     if x[3] > threshold: count_bad +=1
#     else:
#         count_good +=1
# print(count_bad, count_good+count_bad, count_bad/(count_bad+count_good))

# for x in sorted(dist, key=lambda x: x[3], reverse=True)[:30]:
#     print(x)