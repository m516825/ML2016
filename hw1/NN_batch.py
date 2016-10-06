import csv
import argparse
import numpy as np
import sys
import random
import math

def parse_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--iteration', default=50000, type=int)
	parser.add_argument('--learning_rate', default=0.00000005, type=float)
	parser.add_argument('--momentum', default=1, type=int)
	parser.add_argument('--train_data', default='./data/train.csv', type=str)
	parser.add_argument('--test_data', default='./data/test_X.csv', type=str)
	parser.add_argument('--output_file', default='./linear_regression.csv', type=str)
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

def calculate_error(w1, w2, b1, b2, x_dat, y_dat, Neuron):

	size = len(x_dat)
	error = 0.
	for i in range(size):
		# a = w2[0]*(np.dot(x_dat[i], w1[0].T)+b1[0]) + w2[1]*(np.dot(x_dat[i], w1[1].T)+b1[1]) + w2[2]*(np.dot(x_dat[i], w1[2].T)+b1[2]) + b2 
		a = 0
		for n in range(Neuron):
			a += w2[n] * (np.dot(x_dat[i], w1[n].T)+b1[n]) 
		a += b2
		error += (a - y_dat[i])**2
	error /= float(size)

	return np.sqrt(error)

def create_val_data(x_dat, y_dat):

	size = len(x_dat)
	val_size = int(size/10)
	val_x = x_dat[-val_size:]
	val_y = y_dat[-val_size:]
	x_dat = x_dat[:-val_size]
	y_dat = y_dat[:-val_size]

	return x_dat, y_dat, val_x, val_y

def expand_train(x_dat):

	size = len(x_dat[0])
	for i, dat in enumerate(x_dat):
		tmp = []
		p_dat = dat[-18:]
		for i_1 in range(0, 18-1):
			for i_2 in range(i_1+1, 18):
				tmp.append(p_dat[i_1]*p_dat[i_2]*0.001)
		tmp = np.array(tmp)
		x_dat[i] = np.append(x_dat[i], tmp)
		x_dat[i] = np.append(x_dat[i], p_dat*p_dat*0.001)

	return x_dat

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
			batch_x.append(np.array(tmp_x))
			batch_y.append(np.array(tmp_y))
			tmp_x = []
			tmp_y = []
	if tmp_x != []:
		batch_x.append(np.array(tmp_x))
		batch_y.append(np.array(tmp_y))

	return batch_x, batch_y, batch_number

def train(args, x_dat, y_dat):

	x_dat = expand_train(x_dat)
	x_dat, y_dat, val_x, val_y = create_val_data(x_dat, y_dat)
	batch_x, batch_y, batch_number = make_batch(x_dat, y_dat, 20)

	train_size = len(x_dat)
	f_size = len(x_dat[0])
	print f_size
	Neuron = 5
	w1 = np.random.uniform(-.01, .01, (Neuron, f_size))
	w2 = np.random.uniform(-.01, .01, (Neuron))
	b1 = np.zeros((Neuron))
	b2 = 0.
	gradsq_w1 = np.zeros((Neuron, f_size)) + 1.
	gradsq_b1 = np.zeros((Neuron)) + 1.

	gradsq_w2 = np.zeros((Neuron)) + 1.
	gradsq_b2 = 1.
	cost = 0.
	Lambda = 0
	eta = args.learning_rate
	viol = 0

	pre_eout = float('Inf')

	for iters in range(args.iteration):
		cost = 0.
		for i, b_dat in enumerate(batch_x):
			# diff = np.dot(b_dat, w.T).reshape((len(b_dat), 1)) + b - batch_y[i]
			# cost += np.sum(0.5 * diff * diff) + 0.5 * Lambda * np.sum(w**2) * len(b_dat)

			# w -= eta * (np.sum(diff * b_dat, axis=0) + Lambda * w * len(b_dat)) / np.sqrt(gradsq_w)
			# b -= eta * np.sum(diff) / math.sqrt(gradsq_b)

			# gradsq_w += (eta * (np.sum(diff * b_dat, axis=0) + Lambda * w * len(b_dat)))**2
			# gradsq_b += eta * np.sum(diff) * eta * np.sum(diff)
			diff = 0
			for n in range(Neuron):
				diff += w2[n] * (np.dot(b_dat, w1[n].T).reshape((len(b_dat), 1))+b1[n]) 
			diff = diff + b2 - y_dat[i]
			cost += np.sum(0.5 * diff * diff) 

			_w2 = np.sum(diff * (np.dot(w1, b_dat.T).T + b1), axis=0) #+ Lambda * w2
			_b2 = np.sum(diff)
			_w1 = np.sum(diff * b_dat, axis=0) * w2[None,:].T #+ Lambda * w1
			_b1 = np.sum(diff * w2, axis=0)

			w1 -= eta * _w1 / np.sqrt(gradsq_w1)
			b1 -= eta * _b1 / np.sqrt(gradsq_b1)
			w2 -= eta * _w2 / np.sqrt(gradsq_w2)
			b2 -= eta * _b2 / np.sqrt(gradsq_b2)

			gradsq_w1 += eta * _w1 * eta * _w1
			gradsq_b1 += eta * _b1 * eta * _b1
			gradsq_w2 += eta * _w2 * eta * _w2
			gradsq_b2 += eta * _b2 * eta * _b2

		ein = calculate_error(w1, w2, b1, b2, x_dat, y_dat, Neuron)
		eout = calculate_error(w1, w2, b1, b2, val_x, val_y, Neuron)
		print >> sys.stderr, 'iters '+str(iters)+', cost >> '+str(cost/float(train_size))+', ein '+str(ein)+', eout '+str(eout)

		if eout < pre_eout:
			pre_eout = eout
			viol = 0
		else:
			viol += 1
			if iters > 300 and viol > 10:
				# break
				print viol
			else:
				pre_eout = eout

	return w1, w2, b1, b2, Neuron

def test(w1, w2, b1, b2, t_x_dat, Neuron):

	t_x_dat = expand_train(t_x_dat)
	ans = []
	for dat in t_x_dat:
		# a = w2[0]*(np.dot(dat, w1[0].T)+b1[0]) + w2[1]*(np.dot(dat, w1[1].T)+b1[1]) + w2[2]*(np.dot(dat, w1[2].T)+b1[2]) + b2 
		a = 0
		for n in range(Neuron):
			a += w2[n] * (np.dot(dat, w1[n].T)+b1[n]) 
		a += b2
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

	w1, w2, b1, b2, Neuron = train(args, x_dat, y_dat)

	ans = test(w1, w2, b1, b2, t_x_dat, Neuron)

	output_ans(args, ans)

if __name__ == '__main__':
	main()