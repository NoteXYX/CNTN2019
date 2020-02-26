import args
import chainer.links as L
import chainer
import data_handler as dh
import model as cntn
import numpy as np
from chainer import Chain, optimizers, serializers, Variable
from util import key2value
import json
from getkp import getkp

###load arguments
arg = args.process_command()
testing_url = arg.predict
doc_len = arg.dlen
word_len = arg.wlen
word_dim = arg.wdim
n_units = arg.hdim
n_label = arg.label
filter_length = arg.flen
filter_width = word_len
filter_height = word_dim
output_channel = arg.channel
batch_size = arg.batch
n_epoch = arg.epoch
model_url = arg.model


def loadLine(line, doc_len, word_len):
	dataset = dh.load_corpus(line, doc_len, word_len)
	x = dataset['source']
	keyword = dataset['keyword']
	org_word = dataset['org']
	return x, keyword, org_word

###predict
def predict(x, keyword, model):
	print('###\tpredict')
	learned_y = []
	slen = lambda a, b, c: c if a-b > c else a-b
	N = len(keyword)
	for i in range(0,  N, batch_size ):
		_ = slen(N, i, batch_size)
		x = chainer.Variable(np.asarray(x).astype(np.float32)).reshape(-1, 1, doc_len, word_len, word_dim)
		w = chainer.Variable(np.asarray(keyword[i:i+_]).astype(np.float32)).reshape(-1, 1, doc_len, word_len, word_dim)
		y = model(x, w)
		learned_y.extend(y.data)
	return learned_y

###load oldmodel
print ('###\tload model\t:{}'.format( model_url ))
print ('###\tpredicted txt\t:{}'.format( testing_url ))

model = cntn.CNTN(output_channel, filter_length, filter_width, filter_height, n_units, n_label)
cf = L.Classifier(model)
optimizer = optimizers.Adam()
optimizer.setup(cf)
serializers.load_npz(model_url, model)

###load dataset
testFile = open(testing_url, 'r', encoding='utf-8')
tests = testFile.readlines()
preNum = 0.0
groundNum = 0.0
goodNum = 0.0
for test in tests:
	jsonData = json.load(test)
	line = jsonData["abstract"].strip().lower()
	curPreKw = []
	x, keyword, org_word = loadLine(line, doc_len, word_len)
	learned_y = predict(x, keyword, model)
	predicted = [np.argmax(learned_y[i]) for i in range(len(learned_y))]
	for score, word in zip(predicted, org_word):
		if score == 1:
			curPreKw.append(word)
	curPreKp = getkp(line, curPreKw)
	curGroundKp = jsonData["keywords"].split(';')
	groundNum += len(curGroundKp)
	preNum += len(curPreKp)
	for phrase in curPreKp:
		if phrase in curGroundKp:
			goodNum += 1

print('---\toutput result\t\t---')
precision = goodNum / preNum
recall = goodNum / groundNum
f1 = 2 * precision * recall / (precision + recall)
print('dataset: {}'.format(testing_url))
print('model:{}'.format(model_url))
print('precision:{:.2f}, recall:{:.2f}, F1-score:{:.2f}'.format(precision, recall, f1))
with open('result/predict.txt', 'w', encoding='utf-8') as fres:
	fres.write('------------------------------------\n')
	fres.write('dataset: {}\n'.format(testing_url))
	fres.write('model:{}\n'.format(model_url))
	fres.write('precision:{:.2f}, recall:{:.2f}, F1-score:{:.2f}\n'.format(precision, recall, f1))








###write file	
# print('###\toutput Keywords\t:{}'.format(testing_url+'.key'))
# predicted = [np.argmax(learned_y[i]) for i in range(len(learned_y))]
# with open( testing_url+'.key', 'w' ) as f:
# 	for i in zip(predicted, org_word):
# 		if i[0] == 1:
# 			f.write('{}\n'.format(i[1]))
