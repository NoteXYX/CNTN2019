import json
import nltk
import util


def main(token='_'):
    fin_name = 'data/kp20k/kp20k_train.json'
    fin = open(fin_name, 'r', encoding='utf-8')
    fout = open('data/kp20k/mytrain.txt', 'w', encoding='utf-8')
    i = 0
    for line in fin.readlines():
        jsonData = json.loads(line)
        cur_ab = jsonData['abstract'].strip().lower()
        wordList = nltk.word_tokenize(cur_ab)
        wordList = list(filter(None, list(map(util.removeStop, wordList))))
        cur_doc = ' '.join(wordList)
        if fin_name == 'data/kp20k/kp20k_train.json':
            cur_kps = jsonData['keywords']
        else:
            cur_kps = jsonData['keywords'].split(';')
        cur_kws = []
        for kp in cur_kps:
            kwsList = kp.split(' ')
            cur_kws.extend(kwsList)
        for i in range(len(cur_kws)):
            fout.write(cur_doc + token + cur_kws[i] + token + '1\n')
        wordNum = 0
        notKeyNum = 0
        while wordNum < len(wordList) and notKeyNum < len(cur_kws):
            if wordList[wordNum] not in cur_kws:
                fout.write(cur_doc + token + wordList[wordNum] + token + '0\n')
                notKeyNum += 1
            wordNum += 1
        i += 1
        print('第%d行数据写入完毕......' % i)
    fin.close()
    fout.close()
if __name__ == '__main__':
    main(token='##:')