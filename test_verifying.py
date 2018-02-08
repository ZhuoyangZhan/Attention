#Performance
limit_num = time_step
hit = 0
index = 0
cache = []

for index in range(len(mc_cut)):
    num = random.randint(0,len(mc_cut)-1)
    while num in cache:
        num = random.randint(0,len(mc_cut)-1)
    item = mc_cut[num]
    cache.append(num)
    if len(item) < limit_num:
        while len(item) < limit_num:
            item.append(('PAD',1))
    elif len(item) > limit_num:
        item = item[:limit_num]
    mc_vec = []    
    for word_tuple in item:
        if (word_tuple[0] != 'PAD'):
            word = word_tuple[0]
            pos = word_tuple[1]
            dep = word_tuple[2]
            try:
                idf = idf_dict[word]
            except:
                idf = 6.0
            try:
                w_vec = w2v[word].astype('float16')
                #pos
                pos_vec = zeros(len(pos_dict)).astype('float16')
                try:
                    pos = pos_dict[pos]
                except:
                    pass
                pos_vec[pos] = 1

                #dep
                dep_vec = zeros(len(dep_dict)).astype('float16')
                try:
                    dep = dep_dict[dep]
                except:
                    pass
                dep_vec[dep] = 1
                word_vec = np.concatenate((w_vec,pos_vec,dep_vec), axis=0)

            except Exception as E:
                word_vec = zeros(300+28+14).astype('float16')
            mc_vec.append(word_vec)
        else:
            mc_vec.append(zeros(300+28+14).astype('float16'))
    mc_vec = np.array(mc_vec).astype('float16')
    
    if encoder.transform([ipc_trim[num]]) in [i[0] for i in sorted(enumerate(model.predict(np.array([mc_vec]))[0]), key = lambda x:-x[1])[:5]]:
        hit += 1
    hit_rate = hit/(index+1)
    
    if index % 1000 == 0:
        print(index, hit_rate)
