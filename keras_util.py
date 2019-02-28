def compute_class_weight(labels):
    #make class_freq
    class_freq={}
    for x in labels:
        if not (x in class_freq):
            #not already got key
            class_freq[x]=1
        else:
            class_freq[x]+=1
    #make class weight
    class_weight={}
    minfreq = min(v for k,v in class_freq.items())
    for k,v in class_freq.items():
        class_weight[k] = minfreq/v
    return class_weight

def group_for_fit_generator(xs, n):
    i = 0
    out = []
    for x in xs:
        i+=1
        out.append(x)
        if i == n:
            i = 0
            yield out
            out = []
    if out != []:
        yield out
