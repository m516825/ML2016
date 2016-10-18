import numpy as np
import sys
import argparse
import csv

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dat', default='./data/spam_train.csv', type=str)
	parser.add_argument('--test_dat', default='./data/spam_test.csv', type=str)
	parser.add_argument('--output_dat', default='./prediction.csv', type=str)
	parser.add_argument('--model', default='./model.npy', type=str)
	parser.add_argument('--learning_rate', default=1e-4, type=float)
	parser.add_argument('--iteration', default=1000, type=int)
	parser.add_argument('--type', default=1, type=int)
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

def make_batch(x_dat, y_dat, batch_size):

	train_size = len(x_dat)
	batch_number = train_size/batch_size
	batch_number += 1 if train_size%batch_size != 0 else 0
	batch_x = []
	batch_y = []
	tmp_x = []
	tmp_y = []
	for i, dat in enumerate(x_dat):
		tmp_x.append(dat)
		tmp_y.append(y_dat[i])
		if (i+1)%batch_size == 0:
			batch_x.append(np.array(tmp_x))
			batch_y.append(np.array(tmp_y))
			tmp_x = []
			tmp_y = []
	if tmp_x != []:
		batch_x.append(np.array(tmp_x))
		batch_y.append(np.array(tmp_y))

	return batch_x, batch_y, batch_number

def logistic_regression(args, train_x, train_y):

	batch_x, batch_y, batch_number = make_batch(train_x, train_y, 50)
	train_size = len(train_x)
	f_size = len(train_x[0])
	w = np.random.uniform(-1, 1, (f_size))
	b = 0.
	grad_w = np.zeros(f_size) + 1.
	grad_b = 1.
	cost = 0.
	eta = args.learning_rate

	for iters in range(args.iteration):
		cost = 0.
		for i, dat in enumerate(batch_x):

			dot_v = np.dot(w, dat.T) + b
			diff = -(batch_y[i] - sigmoid(dot_v))

			v = sigmoid(dot_v)
			index = np.where(v <= 0.)
			v[index] = 1e-100

			_v = 1 - sigmoid(dot_v)
			_index = np.where(_v <= 0.)
			_v[_index] = 1e-100

			cost += (-(batch_y[i]*np.log(v) + (1 - batch_y[i])*np.log(_v))).sum()

			w_g = np.sum(eta * diff[None,:].T * dat, axis=0)
			b_g = (eta * diff).sum()

			w -= w_g / np.sqrt(grad_w)
			b -= b_g / np.sqrt(grad_b)

			grad_w += w_g**2
			grad_b += b_g**2

		ein = count_ein(w, b, train_x, train_y, train_size)
		print >> sys.stderr, 'Iteration '+str(iters)+', cost : '+str(cost/train_size)+', accurary : '+str(1.-ein)


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

	if args.type == 1:

		train_x, train_y = load_train(args)

		w, b = logistic_regression(args, train_x, train_y)

		out = np.append(w, b)

		np.save(args.model, out)

	else:

		if not args.model.endswith('.npy'):
			args.model += '.npy'

		model = np.load(args.model)

		test_x = load_test(args)

		w, b = model[:-1], model[-1]

		ans = ans_test(test_x, w, b)

		dump_ans(args, ans)

if __name__ == '__main__':

	main()