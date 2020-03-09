import argparse

def process_command():
	parser = argparse.ArgumentParser(prog='Training', description='Arguments')
	parser.add_argument('--dlen', '-dl', default=300, help='doc len', type=int )
	parser.add_argument('--wlen', '-wl', default=30, help='word len', type=int )
	parser.add_argument('--wdim', '-wd', default=27, help='word dim', type=int )
	parser.add_argument('--hdim', '-hd', default=50, help='hidden dim', type=int )
	parser.add_argument('--label', '-l', default=2, help='output label', type=int )
	parser.add_argument('--flen', '-fl', default=3, help='filter length', type=int )
	parser.add_argument('--channel', '-c', default=50, help='channel size', type=int )
	parser.add_argument('--batch', '-b', default=300, help='batch size', type=int )	# 64
	parser.add_argument('--epoch', '-e', default=5, help='epoch', type=int )	# 25
	parser.add_argument('--gpu', '-g', default=-1, help='-1=cpu, 0, 1,...= gpt', type=int)
	parser.add_argument('--model', '-model', default='./mykrapivin_model', help='path of model')
	parser.add_argument('--train', '-train', default='./data/krapivin/mytrain.txt', help='path of training data')
	parser.add_argument('--predict', '-predict', default='./data/krapivin/krapivin_test.json', help='path of predicted data')
	

	return parser.parse_args()
