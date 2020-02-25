def getkp(curDoc, keywords):
    kp = []
    for i in range(len(keywords)):
        curkw = keywords[i]
        curkp = []
        j = 0
        findStart = 0
        while j < len(curDoc):
            if findStart == 0:
                if curDoc[j] != curkw:
                    j += 1
                    continue
                else:
                    curkp.append(curkw)
                    findStart = 1
                    j += 1
                    continue
            if curDoc[j] in keywords:
                curkp.append(curDoc[j])
                if j + 1 == len(curDoc) and len(curkp) > 1:
                    tmp = list(set(curkp))
                    tmp.sort(key=curkp.index)
                    curkp = tmp
                    kp.append(' '.join(curkp))
                    curkp = []
                    findStart = 0
            elif len(curkp) > 1:
                tmp = list(set(curkp))
                tmp.sort(key=curkp.index)
                curkp = tmp
                kp.append(' '.join(curkp))
                curkp = []
                findStart = 0
            elif len(curkp) == 1:
                curkp = []
                findStart = 0
            j += 1
    res = list(set(kp))[:5]
    res.sort(key = lambda i: len(i), reverse=True)
    res.extend(keywords)
    wordNum1 = 0
    wordNum2 = 1
    while wordNum1 < len(res) - len(keywords) - 1:
        while wordNum2 < len(res) - len(keywords):
            if repeatNum(res[wordNum1].split(), res[wordNum2].split()) > 1:
                tmp = res.pop(wordNum2)
            else:
                wordNum2 += 1
        wordNum1 += 1
    return res

def repeatNum(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    maxstr = s1
    substr_maxlen = max(len(s1), len(s2))
    for sublen in range(substr_maxlen, -1, -1):
        for i in range(substr_maxlen - sublen + 1):
            if ' '.join(maxstr[i:i + sublen]) in ' '.join(s2):
                return len(maxstr[i:i + sublen])

# mystr = ['i', 'am', 'xyx', 'are', 'you', 'ok']
# key = ['xyx', 'am', 'i', 'you', 'ok']
# print(getkp(mystr, key))
# print(repeatNum('i am xyx 2'.split(), 'i am xyx'.split()))
