import re
import sys
import time
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
stop.add('_')

def normalizeString(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r" : ", ":", string)
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " ( ", string) 
	string = re.sub(r"\)", " ) ", string) 
	string = re.sub(r"\?", " ? ", string) 
	string = re.sub(r"\s{2,}", " ", string)   
	return string.strip().lower()


def removeStop(word):
	if word not in stop:
		return word


def key2value(target, dic):
	return dic[target]

def getkp(curDoc, keywords):
	kp = []
	for i in range(len(keywords)):
		curkw = keywords[i]
		curkp = []
		j = 0
		while j < len(curDoc):
			if curDoc[j] == curkw:
				curkp.append(curkw)
				j += 1
				continue
			if curDoc[j] in keywords:
				curkp.append(curDoc[j])
			elif len(curkp) != 0:
				curkp = list(set(curkp))
				kp.append(' '.join(curkp))
				curkp = []
			j += 1
	return kp

# doc = ['hello', 'i', 'am', 'XYX']
# print(list(map(removeStop, doc)))
