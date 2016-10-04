import csv
import argparse
import numpy as np
import sys
import random
import math

def parse_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--iteration', default=10000, type=int)
	parser.add_argument('--learning_rate', default=0.000001, type=float)
	parser.add_argument('--train_data', default='./data/train.csv', type=str)
	parser.add_argument('--test_data', default='./data/test_X.csv', type=str)
	parser.add_argument('--output_file', default='./output.csv', type=str)
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

def calculate_error(w, b, x_dat, y_dat):

	size = len(x_dat)
	error = 0.
	for i in range(size):
		a = np.dot(w.T, x_dat[i]) + b
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

def train(args, x_dat, y_dat):

	x_dat, y_dat, val_x, val_y = create_val_data(x_dat, y_dat)

	train_size = len(x_dat)
	f_size = len(x_dat[0])
	w = np.random.uniform(-.01, .01, (f_size))
	b = 0.
	gradsq_w = np.array([1.]*f_size)
	gradsq_b = 1.
	cost = 0.
	Lambda = 0.1
	eta = args.learning_rate

	pre_eout = float('Inf')

	for iters in range(args.iteration):
		cost = 0.
		for i, dat in enumerate(x_dat):
			diff = np.dot(w.T, dat) + b - y_dat[i]
			cost += 0.5 * diff * diff + 0.5 * Lambda * np.sum(w**2)

			w -= eta * (diff * dat + Lambda * w) / np.sqrt(gradsq_w)
			b -= eta * diff / math.sqrt(gradsq_b)

			gradsq_w += eta * (diff * dat + Lambda * w) * eta * (diff * dat + Lambda * w)
			gradsq_b += eta * diff * eta * diff

		ein = calculate_error(w, b, x_dat, y_dat)
		eout = calculate_error(w, b, val_x, val_y)
		print >> sys.stderr, 'iters '+str(iters)+', cost >> '+str(cost/float(train_size))+', ein '+str(ein)+', eout '+str(eout)

		if eout < pre_eout:
			pre_eout = eout
		else:
			if iters > 300:
				break
			else:
				pre_eout = eout

	return w, b

def test(w, b, t_x_dat):

	ans = []
	for dat in t_x_dat:
		a = np.dot(w.T, dat) + b if np.dot(w.T, dat) + b > 0. else 0.
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

	w, b = train(args, x_dat, y_dat)

	ans = test(w, b, t_x_dat)

	output_ans(args, ans)

if __name__ == '__main__':
	main()