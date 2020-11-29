import args
import chainer.links as L
import chainer
import data_handler as dh
import model as cntn
import numpy as np
from chainer import Chain, optimizers, serializers, Variable
from util import key2value
import json
from getkp import getkp, getSingleAndMoreKP

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
topk = 10

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
testFile.close()
preNum = 0.0
groundNum = 0.0
goodNum = 0.0
singlePreNum = 0.0
morePreNum = 0.0
singleGroundNum = 0.0
moreGroundNum = 0.0
singleGoodNum = 0.0
moreGoodNum = 0.0
for test in tests:
	jsonData = json.loads(test)
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
	curPreSingleKP, curPreMoreKP = getSingleAndMoreKP(curPreKp)
	curGroundSingleKP, curGroundMoreKP = getSingleAndMoreKP(curGroundKp)
	groundNum += len(curGroundKp)
	singleGroundNum += len(curGroundSingleKP)
	moreGroundNum += len(curGroundMoreKP)
	num = min(len(curPreKp), topk)
	singleNum = min(len(curPreSingleKP), topk)
	moreNum = min(len(curPreMoreKP), topk)
	preNum += num
	singlePreNum += singleNum
	morePreNum += moreNum
	for phrase in curPreKp:
		if phrase in curGroundKp:
			goodNum += 1
	for phrase in curPreSingleKP:
		if phrase in curGroundSingleKP:
			singleGoodNum += 1
	for phrase in curPreMoreKP:
		if phrase in curGroundMoreKP:
			moreGoodNum += 1

print('---\toutput result\t\tTop%d---' % topk)
precision = goodNum / preNum
recall = goodNum / groundNum
f1 = 2 * precision * recall / (precision + recall)
singleF1 = 0.0
if singlePreNum != 0 and singleGroundNum != 0:
	singlePrecision = singleGoodNum / singlePreNum
	singleRecall = singleGoodNum / singleGroundNum
	singleF1 = 2 * singlePrecision * singleRecall / (singlePrecision + singleRecall)
moreF1 = 0.0
if morePreNum != 0 and moreGroundNum != 0:
	morePrecision = moreGoodNum / morePreNum
	moreRecall = moreGoodNum / moreGroundNum
	moreF1 = 2 * morePrecision * moreRecall / (morePrecision + moreRecall)
print('dataset: {}'.format(testing_url))
print('model:{}'.format(model_url))
print('precision:{:.4f}, recall:{:.4f}, F1-score:{:.4f}, singleF1:{:.4f}, moreF1:{:.4f}\n'.format(precision, recall, f1, singleF1, moreF1))
with open('result/predictNEW.txt', 'a', encoding='utf-8') as fres:
	fres.write('--------------Top%d----------------------\n' % topk)
	fres.write('dataset: {}\n'.format(testing_url))
	fres.write('model:{}\n'.format(model_url))
	fres.write('precision:{:.4f}, recall:{:.4f}, F1-score:{:.4f}, singleF1:{:.4f}, moreF1:{:.4f}\n'.format(precision, recall, f1, singleF1, moreF1))









###write file	
# print('###\toutput Keywords\t:{}'.format(testing_url+'.key'))
# predicted = [np.argmax(learned_y[i]) for i in range(len(learned_y))]
# with open( testing_url+'.key', 'w' ) as f:
# 	for i in zip(predicted, org_word):
# 		if i[0] == 1:
# 			f.write('{}\n'.format(i[1]))
