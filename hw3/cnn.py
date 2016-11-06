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
flags.DEFINE_integer('iterations', 250, 'Total training iterations')
flags.DEFINE_boolean('interactive', False, 'If true, enters an IPython interactive session to play with the trained')
flags.DEFINE_integer('batch_size', 100, 'batch size for training')
FLAGS = flags.FLAGS

class Data(object):

	def __init__(self, label_dat, unlabel_dat, test_dat):

		self.label_data_x, self.label_data_y, self.n_class = self.format_train_data(label_dat)
		self.split_valid(valid=150)
		self.test_dat_x = self.format_test_data(test_dat)
		self.train_x, self.train_y = self.label_data_x, self.label_data_y
		self.undabel_x, self.undabel_y = self.format_unlabel_data(unlabel_dat), None 
		self.train_size = len(self.label_data_y)
		self.current = 0

	def split_valid(self, valid):
		zip_data = list(zip(self.label_data_x, self.label_data_y))
		random.shuffle(zip_data)
		self.label_data_x, self.label_data_y = zip(*zip_data)
		self.valid_data_x, self.valid_data_y = np.array(self.label_data_x[-valid:]), np.array(self.label_data_y[-valid:])
		self.label_data_x, self.label_data_y = np.array(self.label_data_x[:-valid]), np.array(self.label_data_y[:-valid])

	def format_unlabel_data(self, unlabel_dat):

		undabel_x = []
		for dat in unlabel_dat:
			undabel_x.append(dat)

		return np.array(undabel_x)

	def format_test_data(self, test_dat):

		label_data_x = []
		for dat in test_dat['data']:
			label_data_x.append(dat)

		return np.array(label_data_x)

	def format_train_data(self, label_dat):

		label_data_x = []
		label_data_y = []
		for cat in range(len(label_dat)):
			for t in range(len(label_dat[cat])):
				label_data_x.append(label_dat[cat][t]) # reshape data to 32*32*3
				label_data_y.append(int(cat))

		return np.array(label_data_x), np.array(label_data_y), len(label_dat)

	def next_batch(self, size):
		batch_x = batch_y = None

		if self.current == 0:
			zip_data = list(zip(self.train_x, self.train_y))
			random.shuffle(zip_data)
			self.train_x, self.train_y = zip(*zip_data)

		if self.current + size < self.train_size:
			batch_x, batch_y = self.train_x[self.current : self.current+size], self.train_y[self.current : self.current+size]
			self.current += size
		else:
			batch_x, batch_y = self.train_x[self.current :], self.train_y[self.current :]
			self.current = 0
		return batch_x, batch_y

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--label_dat', default='./data/all_label.p', type=str)
	parser.add_argument('--unlabel_dat', default='./data/all_unlabel.p', type=str)
	parser.add_argument('--test_dat', default='./data/test.p', type=str)
	parser.add_argument('--output', default='./pred.csv', type=str)
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

def cnn_model(args, data, mtype, self_train, last=False):

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
			threshod = 0.1
			for ite in range(FLAGS.iterations):

				batch_number = data.train_size/FLAGS.batch_size
				batch_number += 1 if data.train_size%FLAGS.batch_size !=0 else 0
				cost = 0.

				pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_number).start()
				print >> sys.stderr, 'Self-train {}, Iterations {} :'.format(self_train, ite)

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

				print >> sys.stderr, '>>> cost: {}, acc: {}, v_acc: {}'.format(cost, accuracy, v_accuracy)

				if (accuracy-v_accuracy > threshod and accuracy > 0.85 or ite == FLAGS.iterations-1) and not last and len(data.undabel_x) > 0:
					print >> sys.stderr, 'assign unlabel...'
					np.random.shuffle(data.undabel_x)
					soft_max_out = sess.run(pred_con, feed_dict={train_x:data.undabel_x, p_keep_dens:1.0})

					unlabel_list = []
					label_list_x = []
					label_list_y = []
					pred_max = np.amax(soft_max_out, axis=1)
					args_max = np.argmax(soft_max_out, axis=1)
					adding = 0
					topk = accuracy*2000
					for pred, _x, label in sorted(zip(pred_max, data.undabel_x, args_max), key=lambda pair:pair[0], reverse=True):
						if pred > 0.95 and adding <= topk:
							label_list_x.append(_x)
							label_list_y.append(label)
							adding += 1
						else:
							unlabel_list.append(_x)

					print >> sys.stderr, '>>> topk={}, adding {}'.format(topk,adding)
					if len(label_list_x) > 0:
						data.train_x = np.concatenate((data.train_x, np.array(label_list_x)), axis=0)
						data.train_y = np.concatenate((data.train_y, np.array(label_list_y)))
						data.train_size = len(data.train_y)
						data.undabel_x = np.array(unlabel_list)

					break


			# if not os.path.exists("./model"):	
			# 	os.makedirs("./model")
			saver.save(sess, './cnn_%s.ckpt'%args.model)

		else:

			saver.restore(sess, './cnn_%s.ckpt'%args.model)
			pre_result = sess.run(predict_op, feed_dict={train_x:data.test_dat_x, p_keep_dens:1.0})

			with open(args.output, 'w') as f:
				f.write('ID,class\n')
				for i, p in enumerate(pre_result):
					out = str(i)+','+str(p)+'\n'
					f.write(out)

		return data

def main(_):

	args = arg_parse()

	label_dat, unlabel_dat, test_dat = load_data(args)

	data = Data(label_dat, unlabel_dat, test_dat)

	# if args.supervised:
	if args.mtype == 'train':
		self_train_iter = 3
		for i in range(self_train_iter):
			if i != self_train_iter-1:
				data = cnn_model(args, data, mtype='train', self_train=i+1)
			else:
				data = cnn_model(args, data, mtype='train', self_train=i+1, last=True)
	else:
		cnn_model(args, data, mtype='test', self_train=None)

if __name__ == '__main__':
	tf.app.run()