import nltk
import os
import util
import wordMatrix as wm
from collections import Counter, deque
from itertools import chain
from numpy import array, concatenate, zeros

def load_data( articles, doc_len, word_len, token='_' ):
	docs = dict()
	words = dict()

	bodocs = dict()
	bowords = dict()
	indexToDoc = dict()
	
	numOfdocs = 0
	numOfwords = 0
	
	char_dim = len(wm.getBOC())+1
	max_len = doc_len

	source = deque()
	kw = deque()
	feature = deque()
	labels = deque()
	testWords = deque()
	i = 0
	for article in articles:
		_ = article.split(token)
		doc = nltk.word_tokenize(_[0])
		if doc == '':
			continue
		keyword = _[1]
		if keyword == '':
			continue
		labels.append(int(_[2].strip()))
		if max_len > len(doc):
			docVec = list(map(wm.vector, doc))
			docVec = array(docVec)
			dzero = zeros((max_len-len(doc), word_len, char_dim))
			docVec = concatenate((docVec, dzero), axis=0)
		else:
			docVec = list(map(wm.vector, doc[:max_len]))
			docVec = array(docVec)

		if _[0] not in bodocs.keys():
			docs[numOfdocs] = docVec
			bodocs[_[0]] = numOfdocs
			indexToDoc[numOfdocs] = _[0]
			numOfdocs += 1

		wordVec = list(map(wm.vector, keyword))
		wzero = zeros((max_len-len(keyword), word_len, char_dim))
		wordVec = concatenate((wordVec, wzero), axis=0)
		i += 1
		print('{} is ok...'.format(i))
		if _[1] not in bowords:
			words[numOfwords] = wordVec
			bowords[_[1]] = numOfwords
			numOfwords += 1
		source.append(bodocs[_[0]])
		kw.append(bowords[_[1]])
		testWords.append(_[1])
		if i == 1884855:
			break


	del bowords
	del bodocs
	dataset = {}
	dataset['source'] = array(source)
	dataset['keyword'] = array(kw)
	dataset['target'] = array(labels)
	dataset['docs'] = docs
	dataset['words'] = words
	dataset['testWords'] = array(testWords)
	dataset['indexToDoc'] = indexToDoc

	return dataset

def load_corpus(corpus, doc_len, word_len):
	char_dim = len(wm.getBOC())+1
	max_len = doc_len

	doc = corpus
	doc = util.normalizeString(doc)
	### preprocessing
	doc = nltk.word_tokenize(doc)
	doc = list(filter(None, list(map(util.removeStop, doc))))
	###
	words = list(set(doc))
	words.sort(key=doc.index)

	docVec = list(map(wm.vector, doc))
	docVec = array(docVec)
	dzero = zeros((max_len-len(doc), word_len, char_dim))
	docVec = concatenate((docVec, dzero), axis=0)

	wordVecs = deque()

	for word in words:
		wordVec = list(map(wm.vector, word))
		wzero = zeros((max_len-len(word), word_len, char_dim))
		wordVec = concatenate((wordVec, wzero), axis=0)
		wordVecs.append(wordVec)

	dataset = {}
	dataset['source'] = array([docVec]*len(words))
	dataset['keyword'] = array(wordVecs)	
	dataset['org'] = words
	return dataset
