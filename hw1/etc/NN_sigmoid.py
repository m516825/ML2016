import csv
import argparse
import numpy as np
import sys
import random
import math

def parse_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--iteration', default=10000, type=int)
	parser.add_argument('--learning_rate', default=1e-5, type=float)
	parser.add_argument('--momentum', default=1, type=int)
	parser.add_argument('--train_data', default='./data/train.csv', type=str)
	parser.add_argument('--test_data', default='./data/test_X.csv', type=str)
	parser.add_argument('--m', default=0, type=int)
	parser.add_argument('--output_file', default='./NN.csv', type=str)
	args = parser.parse_args()

	return args

def make_train_pair(seq_feature):
	x_dat = []
	y_dat = []
	for i, f in enumerate(seq_feature):
		tmp = np.array([])
		if i%(24*20) >= 9:
			for seq in range(i-9, i):
				tmp = np.append(tmp, seq_feature[seq])
			x_dat.append(tmp)
			y_dat.append([seq_feature[i][9]])

	return x_dat, y_dat

def make_test_pair(seq_feature):

	x_dat = []
	for i, f in enumerate(seq_feature):
		tmp = np.array([])
		if (i+1)%9 == 0:
			for seq in range(i-8, i+1): # from 0 ~ 8
				tmp = np.append(tmp, seq_feature[seq])
			x_dat.append(tmp)

	return x_dat

def shuffle(x_dat, y_dat):

	size = len(x_dat)
	for i in range(size):
		r = random.randrange(0, size)
		tmp_x = x_dat[i]
		tmp_y = y_dat[i]
		x_dat[i] = x_dat[r]
		y_dat[i] = y_dat[r]
		x_dat[r] = tmp_x
		y_dat[r] = tmp_y

	return x_dat, y_dat

def load_train(path):

	raw_data = []
	with open(path, 'r') as f:
		for row in csv.reader(f):
			raw_data.append(row[3:])
	raw_data.pop(0)
	
	seq_feature = []
	day_mat = []
	for i, row in enumerate(raw_data):
		tmp = []
		for v in row:
			num_v = float(v) if v != 'NR' else 0.
			tmp.append(num_v)
		day_mat.append(tmp)
		if (i+1)%18 == 0:
			np_day_mat = np.array(day_mat).T
			day_mat = []
			if seq_feature == []:
				seq_feature = np_day_mat
			else:
				seq_feature = np.append(seq_feature, np_day_mat, axis=0)

	x_dat, y_dat = make_train_pair(seq_feature)

	x_dat, y_dat = shuffle(x_dat, y_dat)
	
	return x_dat, y_dat

def load_test(path):

	raw_data = []
	with open(path, 'r') as f:
		for row in csv.reader(f):
			raw_data.append(row[2:])
	
	seq_feature = []
	day_mat = []
	for i, row in enumerate(raw_data):
		tmp = []
		for v in row:
			num_v = float(v) if v != 'NR' else 0.
			tmp.append(num_v)
		day_mat.append(tmp)
		if (i+1)%18 == 0:
			np_day_mat = np.array(day_mat).T
			day_mat = []
			if seq_feature == []:
				seq_feature = np_day_mat
			else:
				seq_feature = np.append(seq_feature, np_day_mat, axis=0)

	x_dat = make_test_pair(seq_feature)
	
	return x_dat

def sigmoid(x):
	return 1/(1+np.exp(-x))

def calculate_error(w, b, x_dat, y_dat, layer):

	size = len(x_dat)
	a = np.array(x_dat).T
	for l in range(layer):
		z = np.dot(w[l], a) + b[l]
		a = sigmoid(z) if l != layer-1 else z
	error = np.sum((a - np.array(y_dat).T)**2) / float(size)

	return np.sqrt(error)

def create_val_data(x_dat, y_dat):

	size = len(x_dat)
	val_size = int(size/1000)
	val_x = x_dat[-val_size:]
	val_y = y_dat[-val_size:]
	x_dat = x_dat[:-val_size]
	y_dat = y_dat[:-val_size]

	return x_dat, y_dat, val_x, val_y

def feature_scaling(x_dat):

	size = len(x_dat[0])
	mean = np.sum(np.array(x_dat), axis=0) / size
	driva = np.array([0.]*size)
	for i, dat in enumerate(x_dat):
		driva += (dat - mean)**2
	driva /= size
	driva = np.sqrt(driva)

	for i, dat in enumerate(x_dat):
		x_dat[i] = (x_dat[i] - mean)/driva

	return x_dat

def expand_train(x_dat):

	size = len(x_dat[0])

	time = []
	for i in range(9):
		time += [(i+1)] * 18
	time = np.array(time)

	for i, dat in enumerate(x_dat):
		tmp = []
		p_dat = dat[-18:]
		for i_1 in range(0, 18-1):
			for i_2 in range(i_1+1, 18):
				tmp.append(p_dat[i_1]*p_dat[i_2])
		tmp = np.array(tmp)
		x_dat[i] = np.append(x_dat[i], tmp)
		x_dat[i] = np.append(x_dat[i], dat*dat)
		x_dat[i] = np.append(x_dat[i], dat*time)

	x_dat = feature_scaling(x_dat)

	return x_dat

def back_prop(param_w, param_b, param_a, param_z, layer, y_dat, Lambda):
	theta = [0.] * layer
	_w = [0.] * layer
	_b = [0.] * layer
	w_sum = 0
	for l in range(layer):
		w_sum += param_w[l].sum()

	for l in range(layer-1, -1, -1):
		if l == layer-1:
			theta[l] = 1 * (param_a[l+1] - y_dat) + Lambda * w_sum
		else:
			theta[l] = (1 - sigmoid(param_z[l])) * sigmoid(param_z[l]) * np.dot(param_w[l+1].T, theta[l+1])

		_w[l] = np.dot(theta[l], param_a[l].T)
		_b[l] = np.sum(theta[l], axis=1)[None, :].T

	return _w, _b

def make_batch(x_dat, y_dat, batch_size):

	batch_number = len(x_dat)/batch_size
	batch_number += 1 if len(x_dat)%batch_size != 0 else 0
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


def train(args, x_dat, y_dat):

	x_dat = expand_train(x_dat)
	# x_dat, y_dat, val_x, val_y = create_val_data(x_dat, y_dat)
	batch_x, batch_y, batch_number = make_batch(x_dat, y_dat, 100)

	train_size = len(x_dat)
	f_size = len(x_dat[0])
	print f_size
	
	NN = [f_size, 10, 10, 1]
	layer = len(NN)-1
	param_w = []
	param_b = []
	param_grad_w = []
	param_grad_b = []

	for n in range(len(NN)-1):
		w = np.random.uniform(-.01, .01, (NN[n+1], NN[n]))
		b = np.random.uniform(-.0, .0, (NN[n+1], 1))
		gw = np.ones((NN[n+1], NN[n]))
		gb = np.ones((NN[n+1], 1))
		param_w.append(w)
		param_b.append(b)
		param_grad_w.append(gw)
		param_grad_b.append(gb)

	cost = 0.
	m_lambda = 0.6
	Lambda = 0
	eta = args.learning_rate

	for iters in range(args.iteration):
		v_w = [0.] * layer
		v_b = [0.] * layer
		cost = 0.
		for i, dat in enumerate(batch_x):

			param_a = []
			param_z = []
			# a = dat[None, :].T
			a = dat
			param_a.append(a)
			for l in range(layer):
				z = np.dot(param_w[l], a) + param_b[l]
				a = sigmoid(z) if l != layer-1 else z
				param_a.append(a)
				param_z.append(z)

			diff = a - batch_y[i]
			cost += np.sum(0.5 * diff * diff)

			_w, _b = back_prop(param_w, param_b, param_a, param_z, layer, batch_y[i], Lambda) 
			
			for l in range(layer):

				if args.m == 1:
					v_w[l] = m_lambda * v_w[l] - eta * _w[l]
					v_b[l] = m_lambda * v_b[l] - eta * _b[l]
					param_w[l] += v_w[l]
					param_b[l] += v_b[l]

				elif args.m == 2:
					v_w[l] = m_lambda * v_w[l] - eta * _w[l] / np.sqrt(param_grad_w[l])
					v_b[l] = m_lambda * v_b[l] - eta * _b[l] / np.sqrt(param_grad_b[l])

					param_w[l] += v_w[l]
					param_b[l] += v_b[l]

					param_grad_w[l] += (eta * _w[l]) * (eta * _w[l])
					param_grad_b[l] += (eta * _b[l]) * (eta * _b[l])

				else:
					param_w[l] -= eta * _w[l] / np.sqrt(param_grad_w[l])
					param_b[l] -= eta * _b[l] / np.sqrt(param_grad_b[l])

					param_grad_w[l] += eta * _w[l] * eta * _w[l]
					param_grad_b[l] += eta * _b[l] * eta * _b[l]

		ein = calculate_error(param_w, param_b, x_dat, y_dat, layer)
		# eout = calculate_error(param_w, param_b, val_x, val_y, layer)
		print >> sys.stderr, 'iters '+str(iters)+', cost >> '+str(cost/float(train_size))+', ein '+str(ein)#+', eout '+str(eout)

	args.output_file = 'NN_sig, eta['+str(eta)+'],'+'lambda['+str(Lambda)+'],'+'Ein['+str(ein)+'].csv'

	return param_w, param_b, layer

def test(w, b, t_x_dat, layer):

	t_x_dat = expand_train(t_x_dat)
	ans = []
	for dat in t_x_dat:
		# a = w2[0]*(np.dot(dat, w1[0].T)+b1[0]) + w2[1]*(np.dot(dat, w1[1].T)+b1[1]) + w2[2]*(np.dot(dat, w1[2].T)+b1[2]) + b2 
		a = dat[None, :].T
		for l in range(layer):
			z = np.dot(w[l], a) + b[l]
			a = sigmoid(z) if l != layer-1 else z
		a = a if a > 0. else 0.
		ans.append(float(a))

	return ans

def output_ans(args, ans):

	with open(args.output_file, 'w') as f:
		f.write('id,value\n')
		for i, a in enumerate(ans):
			out = 'id_'+str(i)+','+str(a)+'\n'
			f.write(out)

def main():

	args = parse_args()

	x_dat, y_dat = load_train(args.train_data)

	t_x_dat = load_test(args.test_data)

	param_w, param_b, layer = train(args, x_dat, y_dat)

	ans = test(param_w, param_b, t_x_dat, layer)

	output_ans(args, ans)

if __name__ == '__main__':
	main()