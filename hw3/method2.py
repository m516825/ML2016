import numpy as np
import argparse 
import tensorflow as tf
import sys 
import os
import cPickle as pickle 
import random
import progressbar as pb

flags = tf.app.flags
flags.DEFINE_float('lr', 5e-3, 'Initial learning rate')
flags.DEFINE_integer('iterations', 400, 'Total training iterations')
flags.DEFINE_boolean('interactive', False, 'If true, enters an IPython interactive session to play with the trained')
flags.DEFINE_integer('batch_size', 100, 'batch size for training')

flags.DEFINE_float('s_lr', 5e-4, 'Initial learning rate')
flags.DEFINE_integer('s_iterations', 100, 'Total training iterations')
flags.DEFINE_integer('s_batch_size', 100, 'batch size for training')
FLAGS = flags.FLAGS

class Data(object):

	def __init__(self, label_dat, test_dat, unlabel_dat):

		if unlabel_dat != None:
			self.label_data_x, self.label_data_y, self.n_class = self.format_train_data(label_dat[0]), label_dat[1], label_dat[2]
			self.split_valid(valid=150)
			self.train_dat_x, self.train_dat_y = self.format_total_data(unlabel_dat[0], unlabel_dat[1])
			self.train_size = len(self.train_dat_y)
			self.current = 0
		self.test_dat_x = self.format_test_data(test_dat)
		self.n_class = label_dat[2]

	def format_total_data(self, unlabel_dat_x, unlabel_dat_y):
		train_dat_x = []
		train_dat_y = []
		for i, _ in enumerate(unlabel_dat_x):
			train_dat_x.append(unlabel_dat_x[i])
			train_dat_y.append(unlabel_dat_y[i])
		for i, _ in enumerate(self.label_data_x):
			train_dat_x.append(self.label_data_x[i])
			train_dat_y.append(self.label_data_y[i])

		return np.array(train_dat_x), np.array(train_dat_y)

	def split_valid(self, valid):
		zip_data = list(zip(self.label_data_x, self.label_data_y))
		random.shuffle(zip_data)
		self.label_data_x, self.label_data_y = zip(*zip_data)
		self.valid_data_x, self.valid_data_y = self.label_data_x[-valid:], self.label_data_y[-valid:]
		self.label_data_x, self.label_data_y = self.label_data_x[:-valid], self.label_data_y[:-valid]

	def format_test_data(self, test_dat):

		label_data_x = []
		for dat in test_dat:
			label_data_x.append(dat)

		return np.array(label_data_x)#.reshape(-1, 16, 16, 1)

	def format_train_data(self, label_dat):

		label_data_x = []
		for dat in label_dat:
			label_data_x.append(dat) 

		return np.array(label_data_x)#.reshape(-1, 16, 16, 1)

	def next_batch(self, size):
		batch_x = batch_y = None

		if self.current == 0:
			zip_data = list(zip(self.train_dat_x, self.train_dat_y))
			random.shuffle(zip_data)
			self.train_dat_x, self.train_dat_y = zip(*zip_data)

		if self.current + size < self.train_size:
			batch_x, batch_y = self.train_dat_x[self.current : self.current+size], self.train_dat_y[self.current : self.current+size]
			self.current += size
		else:
			batch_x, batch_y = self.train_dat_x[self.current :], self.train_dat_y[self.current :]
			self.current = 0
		return batch_x, batch_y

class Data_semi(object):
	def __init__(self, label_dat, unlabel_dat, test_dat):
		self.total_dat = self.combine_data_toGray(label_dat, unlabel_dat, test_dat)
		self.total_size = len(self.total_dat)
		self.current = 0
		###########
		self.unlabel_dat_x = unlabel_dat
		self.label_dat_x, self.label_data_y, self.n_class = self.format_train_data(label_dat)
		###########
		self.gray_unlabel_dat_x = self.unlabel_toGray(unlabel_dat)
		self.gray_label_data_x = self.label_toGray(label_dat)
		###########
		self.test_dat = [dat for dat in test_dat['data']]
		self.feature_size = len(self.test_dat[0])

	def unlabel_toGray(self, unlabel_dat):

		gray_unlabel_dat_x = []
		for dat in unlabel_dat:
			gray_unlabel_dat_x.append(self.toGray(dat))

		return np.array(gray_unlabel_dat_x)

	def label_toGray(self, label_dat):

		gray_label_data_x = []
		for cat in range(len(label_dat)):
			for t in range(len(label_dat[cat])):
				gray_label_data_x.append(self.toGray(label_dat[cat][t]))

		return np.array(gray_label_data_x)

	def toGray(self, array):

		RGB = np.array(array).reshape([-1, 3, 32, 32])
		gray = (RGB[0][0]*299. + RGB[0][1]*587. + RGB[0][2]*114.+500.) / 1000.
		gray = gray.reshape([-1])

		return gray

	def format_train_data(self, label_dat):

		label_data_x = []
		label_data_y = []
		for cat in range(len(label_dat)):
			for t in range(len(label_dat[cat])):
				label_data_x.append(label_dat[cat][t]) # reshape data to 32*32*3
				label_data_y.append(int(cat))

		return label_data_x, label_data_y, len(label_dat)

	def combine_data_toGray(self, label_dat, unlabel_dat, test_dat):

		total_dat = []
		for cat in range(len(label_dat)):
			for t in range(len(label_dat[cat])):
				total_dat.append(self.toGray(label_dat[cat][t]))

		for dat in unlabel_dat:
			total_dat.append(self.toGray(dat))

		# for dat in test_dat['data']:
		# 	total_dat.append(dat)

		return np.array(total_dat) 

	def next_batch(self, size):

		batch_x = None
		if self.current == 0:
			np.random.shuffle(self.total_dat)

		if self.current + size < self.total_size:
			batch_x = self.total_dat[self.current : self.current+size]
			self.current += size
		else:
			batch_x = self.total_dat[self.current:]
			self.current = 0

		return batch_x

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--label_dat', default='./data/all_label.p', type=str)
	parser.add_argument('--unlabel_dat', default='./data/all_unlabel.p', type=str)
	parser.add_argument('--test_dat', default='./data/test.p', type=str)
	parser.add_argument('--output', default='./auto_pred.csv', type=str)
	parser.add_argument('--supervised', default=False, type=bool)
	parser.add_argument('--model', default='model', type=str)
	parser.add_argument('--mtype', default='train', type=str)
	args = parser.parse_args()

	return args

def load_data(args):

	unlabel_dat = None

	label_dat = pickle.load(open(args.label_dat, 'rb'))
	# print len(label_dat), len(label_dat[0]), len(label_dat[0][0])  10 500 3072
	if not args.supervised:
		unlabel_dat = pickle.load(open(args.unlabel_dat, 'rb'))
	# print len(unlabel_dat), len(unlabel_dat[0])
	test_dat = pickle.load(open(args.test_dat, 'rb'))
	# print test_dat['ID'][0], len(test_dat['data'])

	return label_dat, unlabel_dat, test_dat

def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x ,W, strides=[1, strides, strides,1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def cnn_model(args, data, mtype):

	with tf.Graph().as_default(), tf.Session() as sess:

		train_x = tf.placeholder(tf.float32, [None, 1024*3])
		t_train_x = tf.transpose(tf.reshape(train_x, [-1, 3, 32, 32]), [0,2,3,1])
		train_y = tf.placeholder(tf.int32, [None])

		wc1 = tf.Variable(tf.random_normal([3, 3, 3, 24], stddev=0.01))	# 3*3 filter, 3 channel, 24 output
		bc1 = tf.Variable(tf.random_normal([24]))
		wc2 = tf.Variable(tf.random_normal([3, 3, 24, 48], stddev=0.01))	# 3*3 filter, 24 input, 48 output
		bc2 = tf.Variable(tf.random_normal([48]))
		wc3 = tf.Variable(tf.random_normal([3, 3, 48, 48],stddev=0.01))	# 3*3 filter, 48 input, 96 output
		bc3 = tf.Variable(tf.random_normal([48]))
		wd1 = tf.Variable(tf.random_normal([4*4*48, 256], stddev=0.01))	# Dense 1536 * 512
		bd1 = tf.Variable(tf.random_normal([256]))
		wd_out = tf.Variable(tf.random_normal([256, data.n_class]))	# output layer 512 * class
		bd_out = tf.Variable(tf.random_normal([data.n_class])) 

		p_keep_dens = tf.placeholder(tf.float32)

		# 32*32*3 -> 32*32*24 -> 16*16*24
		lc1_con = conv2d(t_train_x, wc1, bc1)		
		lc1 = maxpool2d(lc1_con)
		lc1 = tf.nn.dropout(lc1, p_keep_dens)
		# 16*16*24 -> 16*16*48 -> 8*8*48
		lc2_con = conv2d(lc1, wc2, bc2)
		lc2 = maxpool2d(lc2_con)
		lc2 = tf.nn.dropout(lc2, p_keep_dens)
		# 8*8*48 -> 8*8*96 -> 4*4*96
		lc3_con = conv2d(lc2, wc3, bc3)
		lc3 = maxpool2d(lc3_con)

		lc3 = tf.reshape(lc3, [-1, wd1.get_shape().as_list()[0]])
		lc3 = tf.nn.dropout(lc3, p_keep_dens)


		ld1 = tf.add(tf.matmul(lc3, wd1), bd1)
		ld1 = tf.nn.relu(ld1)
		ld1 = tf.nn.dropout(ld1, p_keep_dens)

		out = tf.add(tf.matmul(ld1, wd_out), bd_out)

		pred_con = tf.nn.softmax(out)

		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, train_y))
		optimizer = tf.train.AdagradOptimizer(FLAGS.lr).minimize(loss)

		predict_op = tf.argmax(out, 1)

		saver = tf.train.Saver()

		if mtype == 'train':
			tf.initialize_all_variables().run()
			# saver.restore(sess, './model/%s.ckpt'%args.model)
			for ite in range(FLAGS.iterations):

				batch_number = data.train_size/FLAGS.batch_size
				batch_number += 1 if data.train_size%FLAGS.batch_size !=0 else 0
				cost = 0.

				pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_number).start()
				print >> sys.stderr, 'Iterations {} :'.format(ite)

				for b in range(batch_number):
					batch_x, batch_y = data.next_batch(FLAGS.batch_size)
					# print batch_y
					c, _ = sess.run([loss, optimizer], feed_dict={train_x:batch_x, train_y:batch_y, p_keep_dens:0.7})
					cost += c / batch_number

					pbar.update(b+1)
				pbar.finish()

				pre_result = sess.run(predict_op, feed_dict={train_x:data.label_data_x, p_keep_dens:1.0})
				accuracy = np.equal(np.array(data.label_data_y), pre_result).mean()
				print pre_result[:15]
				print np.array(data.label_data_y)[:15]

				v_pre_result = sess.run(predict_op, feed_dict={train_x:data.valid_data_x, p_keep_dens:1.0})
				v_accuracy = np.equal(np.array(data.valid_data_y), v_pre_result).mean()

				print >> sys.stderr, '>>> cost: {}, acc: {:.4f}, v_acc: {:.4f}'.format(cost, accuracy, v_accuracy)

				if accuracy > 0.9:
					break

			# if not os.path.exists("./model"):	
			# 	os.makedirs("./model")
			saver.save(sess, './m2_%s.ckpt'%args.model)

		else:

			saver.restore(sess, './m2_%s.ckpt'%args.model)
			pre_result = sess.run(predict_op, feed_dict={train_x:data.test_dat_x, p_keep_dens:1.0})

			with open(args.output, 'w') as f:
				f.write('ID,class\n')
				for i, p in enumerate(pre_result):
					out = str(i)+','+str(p)+'\n'
					f.write(out)

def assign_label(label_x, label_y, unlabel, raw_unlabel):
	
	from sklearn.neighbors import NearestNeighbors
	import time

	raw_unlabel_x = []
	unlabel_y = []
	unlabel_dict = {}
	s_time = time.time()
	print 'starting!!!!'
	neigh = NearestNeighbors(n_neighbors=7)
	neigh.fit(unlabel) 
	k_neighbors = neigh.kneighbors(label_x, return_distance=False)

	for i, neighbors in enumerate(k_neighbors):
		label = label_y[i]
		for n in neighbors:
			try:
				unlabel_dict[n][label] = unlabel_dict[n].get(label, 0) + 1
			except:
				unlabel_dict[n] = {}
				unlabel_dict[n][label] = 1

	for k, v in unlabel_dict.iteritems():
		label = max(v, key=v.get)
		if v[label] >= 2 and len(v) == 1:
			raw_unlabel_x.append(raw_unlabel[k])
			unlabel_y.append(label)
		
	print '{} sec, adding {} training data'.format((time.time() - s_time), len(unlabel_y))

	return np.array(raw_unlabel_x), np.array(unlabel_y)

def autoencoder(args, data, mtype):
	f_size = data.feature_size/3
	encoded_feature_train = None
	encoded_feature_test = None

	with tf.Graph().as_default(), tf.Session() as sess:

		train_x = tf.placeholder("float", [None, f_size])

		encode_h1 = tf.Variable(tf.random_normal([f_size, 512]))
		encode_h2 = tf.Variable(tf.random_normal([512, 256]))
		decode_h1 = tf.Variable(tf.random_normal([256, 512]))
		decode_h2 = tf.Variable(tf.random_normal([512, f_size]))

		encode_b1 = tf.Variable(tf.random_normal([512]))
		encode_b2 = tf.Variable(tf.random_normal([256]))
		decode_b1 = tf.Variable(tf.random_normal([512]))
		decode_b2 = tf.Variable(tf.random_normal([f_size]))

		# encoder

		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(train_x, encode_h1), encode_b1))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, encode_h2), encode_b2))

		# feature = tf.add(tf.matmul(layer1, encode_h2), encode_b2)
		feature = layer2
		# decoder

		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, decode_h1), decode_b1))
		layer_out = tf.add(tf.matmul(layer3, decode_h2), decode_b2)

		y_pred = layer_out
		y = train_x

		loss = tf.reduce_mean(tf.pow(y - y_pred, 2))
		optimizer = tf.train.RMSPropOptimizer(FLAGS.s_lr).minimize(loss)
		# optimizer = tf.train.AdagradOptimizer(FLAGS.s_lr).minimize(loss)

		saver = tf.train.Saver()

		if mtype == 'train':

			# tf.initialize_all_variables().run()
			saver.restore(sess, './model/%s_encode.ckpt'%args.model)
			for ite in range(FLAGS.s_iterations):

				batch_number = data.total_size/FLAGS.s_batch_size
				batch_number += 1 if data.total_size%FLAGS.s_batch_size !=0 else 0
				cost = 0.

				pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_number).start()
				print >> sys.stderr, 'Iterations {} :'.format(ite)

				for b in range(batch_number):
					batch_x  = data.next_batch(FLAGS.s_batch_size)
					
					c, _ = sess.run([loss, optimizer], feed_dict={train_x:batch_x})
					cost += c / batch_number

					pbar.update(b+1)
				pbar.finish()

				print >> sys.stderr, '>>> cost : {}'.format(cost)

			if not os.path.exists("./model"):	
				os.makedirs("./model")
			saver.save(sess, './model/%s_encode.ckpt'%args.model)

		
		saver.restore(sess, './model/%s_encode.ckpt'%args.model)
		encoded_feature_label = sess.run(feature, feed_dict={train_x: data.gray_label_data_x})
		encoded_feature_unlabel = sess.run(feature, feed_dict={train_x: data.gray_unlabel_dat_x})
		# encoded_feature_test = sess.run(feature, feed_dict={train_x: data.test_dat})
		unlabel_x, unlabel_y = assign_label(encoded_feature_label, data.label_data_y, encoded_feature_unlabel, data.unlabel_dat_x)


		return data.label_dat_x, data.test_dat, (unlabel_x, unlabel_y)

def main(_):

	args = arg_parse()

	label_dat, unlabel_dat, test_dat = load_data(args)

	if args.mtype == 'train':

		data_semi = Data_semi(label_dat, unlabel_dat, test_dat)

		e_train_x, e_test_x, e_unlabel = autoencoder(args, data_semi, mtype='train')

		label_dat_formated = (e_train_x, data_semi.label_data_y, data_semi.n_class)

		data = Data(label_dat_formated, e_test_x, e_unlabel)
		
		cnn_model(args, data, mtype='train')

	else:
		label_dat_formated = (None, None, len(label_dat))
		e_test_x = [dat for dat in test_dat['data']]
		data = Data(label_dat_formated, e_test_x, None)
		cnn_model(args, data, mtype='test')

if __name__ == '__main__':
	tf.app.run()