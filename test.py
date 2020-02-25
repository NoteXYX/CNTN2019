import args
import chainer.links as L
import chainer
import data_handler as dh
import model as cntn
import numpy as np
import sys
from chainer import Chain, optimizers, serializers, Variable
from util import key2value
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

###load dataset
testing = open(testing_url).readlines()
dataset = dh.load_data(testing, doc_len, word_len )

x_test = dataset['source'] # array: [每一行文章的序号]
keyword_train = dataset['keyword'] # array: [每一行单词的序号]
y_test = dataset['target'] # array: [每一行行尾单词的类别 0 or 1]
docs = dataset['docs'] # dict:{文档序号: 文档向量}
words = dataset['words'] # dict:{单词序号: 单词向量}
testWords = dataset['testWords'] # array: [每一行行尾的单词]
indexToDoc = dataset['indexToDoc']

###load model
print ('model:\t\t{}'.format( model_url ))
print ('predicted files:{}'.format( testing_url ))

model = cntn.CNTN(output_channel, filter_length, filter_width, filter_height, n_units, n_label)
cf = L.Classifier(model)
optimizer = optimizers.Adam()
optimizer.setup(cf)

print ('###\tload model')
serializers.load_npz(model_url, model)
N = len(testing)
print ('---\tpredict\t\t---')

learned_y = []

sum_test_loss = 0.0
sum_test_acc = 0.0

slen = lambda a, b, c: c if a-b > c else a-b

for i in range(0, N, batch_size):
	_ = slen(N, i, batch_size)
	x = chainer.Variable(np.asarray(list(map(key2value,  x_test[i:i+_], [docs]*_))).astype(np.float32)).reshape(-1, 1, doc_len, word_len, word_dim)
	w = chainer.Variable(np.asarray(list(map(key2value, keyword_train[i:i+_], [words]*_))).astype(np.float32)).reshape(-1, 1, doc_len, word_len, word_dim)
	t = chainer.Variable(np.asarray(y_test[i:i+batch_size]).astype(np.int32))
		#f = chainer.Variable(np.asarray(f_train[i:i+batch_size]))
	f = []

	
	y= cf(x, w, t)
	y0 = model(x, w)	
	sum_test_loss += float(y.data) * len(t.data)
	sum_test_acc += float(cf.accuracy.data) * len(t.data)
	learned_y.extend(y0.data)
	

#print 'testing mean loss = {}, accuracy = {}'.format(sum_test_loss/len(testing), sum_test_acc/len(testing)) 

print('---\tfin\t\t---')

predicted = [np.argmax(learned_y[i]) for i in range(len(learned_y))]
curDocIndex = x_test[0]
curGroundKw = []
curPreKw = []
curLine = 0
preNum = 0.0
groundNum = 0.0
goodNum = 0.0
while curLine < len(x_test):
	if x_test[curLine] == curDocIndex:
		if y_test[curLine] == 1:
			curGroundKw.append(testWords[curLine])
		if predicted[curLine] == 1:
			curPreKw.append(testWords[curLine])
		curLine += 1
	else:
		curDocIndex = x_test[curLine]
		curGroundKp = getkp(indexToDoc[curDocIndex-1], curGroundKw)
		curPreKp = getkp(indexToDoc[curDocIndex-1], curPreKw)[:5]	# if top add here
		groundNum += len(curGroundKp)
		preNum += len(curPreKp)
		for phrase in curPreKp:
			if phrase in curGroundKp:
				goodNum += 1
		curGroundKw = []
		curPreKw = []
print('---\toutput result\t\t---')
precision = goodNum / preNum
recall = goodNum / groundNum
f1 = 2 * precision * recall / (precision + recall)
print('dataset: {}'.format(testing_url))
print('model:{}'.format(model_url))
print('precision:{:.2f}, recall:{:.2f}, F1-score:{:.2f}'.format(precision, recall, f1))
with open('predicted_score', 'w') as f:
	for i in range(len(predicted)):
		f.write('{}\n'.format(predicted[i]))
with open('result/res.txt', 'w', encoding='utf-8') as fres:
	fres.write('------------------------------------\n')
	fres.write('dataset: {}\n'.format(testing_url))
	fres.write('model:{}\n'.format(model_url))
	fres.write('precision:{:.2f}, recall:{:.2f}, F1-score:{:.2f}\n'.format(precision, recall, f1))




