import numpy as np
import sys
import argparse
import csv

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dat', default='./spam_data/spam_train.csv', type=str)
	parser.add_argument('--test_dat', default='./spam_data/spam_test.csv', type=str)
	parser.add_argument('--output_dat', default='./output.csv', type=str)
	parser.add_argument('--learning_rate', default=1e-4, type=float)
	parser.add_argument('--iteration', default=1000, type=int)
	args = parser.parse_args()

	return args

def load_train(args):

	train_x = []
	train_y = []
	with open(args.train_dat, 'r') as f:
		for row in csv.reader(f):
			x = map(float, row[1:-1])
			train_x.append(x)
			train_y.append(float(row[-1]))

	return np.array(train_x), np.array(train_y)

def load_test(args):

	test_x = []
	with open(args.test_dat, 'r') as f:
		for row in csv.reader(f):
			x = map(float, row[1:])
			test_x.append(x)

	return np.array(test_x)

def sigmoid(x):
	return 1./(1.+np.exp(-x))

def count_ein(w, b, x, y, size):

	pred_prob = sigmoid(np.dot(x, w.T) + b)
	error = 0.
	for i, pred in enumerate(pred_prob):
		a = 1 if pred > 0.5 else 0
		if a != y[i]:
			error += 1.
	error /= size

	return error

def logistic_regression(args, train_x, train_y):

	train_size = len(train_x)
	f_size = len(train_x[0])
	w = np.random.uniform(-.1, .1, (f_size))
	b = 0.
	grad_w = np.zeros(f_size) + 1.
	grad_b = 1.
	cost = 0.
	eta = args.learning_rate

	for iters in range(args.iteration):
		cost = 0.
		for i, dat in enumerate(train_x):

			diff = -(train_y[i] - sigmoid(np.dot(dat, w.T) + b))
			logv = np.log(sigmoid(np.dot(dat, w) + b)) if sigmoid(np.dot(dat, w) + b) > 0. else -1000000000000.
			_logv = np.log(1. - sigmoid(np.dot(dat, w) + b)) if 1. - sigmoid(np.dot(dat, w) + b) > 0. else -1000000000000.
			
			cost += -((train_y[i] * logv) + (1. - train_y[i]) * _logv)

			w -= eta * diff * dat / np.sqrt(grad_w)
			b -= eta * diff / np.sqrt(grad_b)

			grad_w += (eta * diff * dat)**2
			grad_b += (eta * diff)**2

		ein = count_ein(w, b, train_x, train_y, train_size)
		print >> sys.stderr, 'Iteration '+str(iters)+', cost : '+str(cost/train_size)+', ein : '+str(ein)


	return w, b

def ans_test(test_x, w, b):

	pred_prob = sigmoid(np.dot(test_x, w.T) + b)
	ans = []
	for i in pred_prob:
		if i > 0.5:
			ans.append(1)
		else:
			ans.append(0)

	return ans

def dump_ans(args, ans):

	with open(args.output_dat, 'w') as f:
		f.write('id,label\n')
		for i, a in enumerate(ans):
			f.write(str(i+1)+','+str(a)+'\n')

def main():

	args = arg_parse()

	train_x, train_y = load_train(args)

	test_x = load_test(args)

	w, b = logistic_regression(args, train_x, train_y)

	ans = ans_test(test_x, w, b)

	dump_ans(args, ans)

if __name__ == '__main__':

	main()