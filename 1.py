import json
import nltk

def main():
    fin = open('data/semeval_wo_stem/semeval_valid.json', 'r', encoding='utf-8')
    fout = open('data/semeval_wo_stem/mytrain.txt', 'w', encoding='utf-8')
    for line in fin.readlines():
        jsonData = json.loads(line)
        cur_ab = jsonData['abstract'].strip().lower()
        wordList = nltk.word_tokenize(cur_ab)
        cur_doc = ' '.join(wordList)
        cur_kps = jsonData['keywords'].split(';')
        cur_kws = []
        for kp in cur_kps:
            kwsList = kp.split(' ')
            cur_kws.extend(kwsList)
        for i in range(len(cur_kws)):
            fout.write(cur_doc + '_' + cur_kws[i] + '_' + '1\n')
        wordNum = 0
        notKeyNum = 0
        while wordNum < len(wordList) and notKeyNum < len(cur_kws):
            if wordList[wordNum] not in cur_kws:
                fout.write(cur_doc + '_' + wordList[wordNum] + '_' + '0\n')
                notKeyNum += 1
            wordNum += 1
    fin.close()
    fout.close()
if __name__ == '__main__':
    main()