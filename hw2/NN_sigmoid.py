import numpy as np
import sys
import argparse
import csv

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dat', default='./spam_data/spam_train.csv', type=str)
	parser.add_argument('--test_dat', default='./spam_data/spam_test.csv', type=str)
	parser.add_argument('--output_dat', default='./NN_output.csv', type=str)
	parser.add_argument('--learning_rate', default=1e-1, type=float)
	parser.add_argument('--iteration', default=5000, type=int)
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

def count_ein(w, b, x_dat, y_dat, layer):

	size = len(x_dat)
	error = 0.
	a = np.array(x_dat).T
	for l in range(layer):
		z = np.dot(w[l], a) + b[l]
		a = sigmoid(z) 
	
	for i in range(size):
		ans = 1 if a[0][i] > 0.5 else 0
		if ans != y_dat[i]:
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
			batch_x.append(np.array(tmp_x).T)
			batch_y.append(np.array(tmp_y).T)
			tmp_x = []
			tmp_y = []
	if tmp_x != []:
		batch_x.append(np.array(tmp_x).T)
		batch_y.append(np.array(tmp_y).T)

	return batch_x, batch_y, batch_number

def back_prop(param_w, param_b, param_a, param_z, layer, y_dat):
	theta = [0.] * layer
	_w = [0.] * layer
	_b = [0.] * layer
	w_sum = 0
	for l in range(layer):
		w_sum += param_w[l].sum()

	for l in range(layer-1, -1, -1):
		if l == layer-1:
			theta[l] = (1 - sigmoid(param_z[l])) * sigmoid(param_z[l]) * (- y_dat/param_a[l+1] + 1/(1-param_a[l+1]) - y_dat/(1-param_a[l+1]))#(- np.log(param_a[l+1]) + np.log(1. - param_a[l+1]))#(param_a[l+1] - y_dat)
		else:
			theta[l] = (1 - sigmoid(param_z[l])) * sigmoid(param_z[l]) * np.dot(param_w[l+1].T, theta[l+1])

		_w[l] = np.dot(theta[l], param_a[l].T)
		_b[l] = np.sum(theta[l], axis=1)[None, :].T

	return _w, _b

def feature_scaling(x_dat, train_len):

	size = len(x_dat[0])
	mean = np.sum(np.array(x_dat), axis=0) / size
	driva = np.array([0.]*size)
	for i, dat in enumerate(x_dat):
		driva += (dat - mean)**2
	driva /= size
	driva = np.sqrt(driva)

	for i, dat in enumerate(x_dat):
		x_dat[i] = (x_dat[i] - mean)/driva

	return x_dat[:train_len], x_dat[train_len:]

def logistic_regression(args, train_x, train_y):

	batch_x, batch_y, batch_number = make_batch(train_x, train_y, 50)
	train_size = len(train_x)
	f_size = len(train_x[0])
	
	NN = [f_size, 5, 5, 1]
	layer = len(NN)-1
	param_w = []
	param_b = []
	param_grad_w = []
	param_grad_b = []

	for n in range(len(NN)-1):
		w = np.random.uniform(-.1, .1, (NN[n+1], NN[n]))
		b = np.random.uniform(-.0, .0, (NN[n+1], 1))
		gw = np.ones((NN[n+1], NN[n]))
		gb = np.ones((NN[n+1], 1))
		param_w.append(w)
		param_b.append(b)
		param_grad_w.append(gw)
		param_grad_b.append(gb)

	cost = 0.
	eta = args.learning_rate

	for iters in range(args.iteration):
		cost = 0.
		for i, dat in enumerate(batch_x):

			param_a = []
			param_z = []

			a = dat
			param_a.append(a)
			for l in range(layer):
				
				z = np.dot(param_w[l], a) + param_b[l]
				a = sigmoid(z) 
				param_a.append(a)
				param_z.append(z)

			# diff = -(batch_y[i] - a)
			cost += (-(batch_y[i]*np.log(a) + (1.-batch_y[i])*np.log(1-a))).sum()#np.sum(0.5 * diff * diff)

			_w, _b = back_prop(param_w, param_b, param_a, param_z, layer, batch_y[i]) 

			for l in range(layer):

				param_w[l] -= eta * _w[l] / np.sqrt(param_grad_w[l])
				param_b[l] -= eta * _b[l] / np.sqrt(param_grad_b[l])

				param_grad_w[l] += eta * _w[l] * eta * _w[l]
				param_grad_b[l] += eta * _b[l] * eta * _b[l]

		ein = count_ein(param_w, param_b, train_x, train_y, layer)
		print >> sys.stderr, 'Iteration '+str(iters)+', cost : '+str(cost/train_size)+', accuracy : '+str(1-ein)


	return param_w, param_b, layer

def ans_test(test_x, w, b, layer):

	ans = []
	size = len(test_x)
	a = np.array(test_x).T
	for l in range(layer):
		z = np.dot(w[l], a) + b[l]
		a = sigmoid(z) 
	
	for i in range(size):
		pred_ans = 1 if a[0][i] > 0.5 else 0
		ans.append(pred_ans)

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

	train_x, test_x = feature_scaling(np.append(train_x, test_x, axis=0), len(train_x))

	w, b, layer = logistic_regression(args, train_x, train_y)

	ans = ans_test(test_x, w, b, layer)

	dump_ans(args, ans)

if __name__ == '__main__':

	main()