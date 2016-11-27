import sys
import os
import argparse
import csv
import math
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.cluster import KMeans
from nltk.stem.porter import PorterStemmer
import progressbar as pb
import tensorflow as tf
from sklearn import linear_model
from re import finditer
from sklearn.metrics.pairwise import cosine_similarity

# def new_euclidean_distance(X, Y=None, Y_norm_squared=None, squared=False):
# 	return 1. - cosine_similarity(X,Y)

# from sklearn.cluster import k_means_
# k_means_.euclidean_distance = new_euclidean_distance

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--title_doc', default='./data/title_StackOverflow.txt', type=str)
	parser.add_argument('--pred_pairs', default='./data/check_index.csv', type=str)
	parser.add_argument('--content', default='./data/docs.txt', type=str)
	parser.add_argument('--pred_out', default='./pred_w2v.csv', type=str)
	parser.add_argument('--contextword', default='./skip_gram', type=str)
	parser.add_argument('--model', default='model_un', type=str)
	parser.add_argument('--mode', default='train', type=str)
	args = parser.parse_args()

	return args

class Data(object):

	def __init__(self, train_pair):
		self.train = train_pair
		self.current = 0
		self.length = len(train_pair)

	def next_batch(self, size):
		if self.current == 0:
			np.random.shuffle(self.train)

		if self.current + size < self.length:
			train_x, train_y = self.train[self.current:self.current+size, 0], self.train[self.current:self.current+size, 1][:, None]
			self.current += size

			return train_x, train_y

		else:
			train_x, train_y = self.train[self.current:, 0], self.train[self.current:, 1][:, None]
			self.current = 0

			return train_x, train_y

class Word_Struct(object):

	def __init__(self, title_data):
		self.title_data = title_data
		self.title_dict = []
		self.doc_size = len(title_data)
		###############################
		self.vocab = None
		self.v2i = {}
		self.i2v = {}
		self.vocab_size = None
		###############################
		self.doc_vec = None
		self.df = None

	def build_vocab(self):
		self.vocab = {}
		for i, d in enumerate(self.title_data):
			title_dict = {}
			for word in d:
				self.vocab[word] = self.vocab.get(word, 0) + 1
				title_dict[word] = title_dict.get(word, 0) + 1
			self.title_dict.append(title_dict)

		index = 0
		for w, c in sorted(self.vocab.iteritems(), key=lambda x:x[1], reverse=True):
			self.v2i[w] = index
			self.i2v[index] = w
			index += 1

		self.vocab_size = len(self.vocab)

	def dump_vocab(self):

		with open('./vocab.txt', 'w') as f:
			for w, c in sorted(self.vocab.iteritems(), key=lambda x:x[1], reverse=True): 
				f.write(str(w)+' '+str(c)+'\n')

	def count_df(self):
		self.df = {}
		for d in self.title_data:
			checked = {}
			for word in d:
				if checked.get(word, False) != True:
					vid = self.v2i[word]
					self.df[vid] = self.df.get(vid, 0) + 1
					checked[word] = True

	def count_doc_vec(self):
		self.doc_vec = []
		for i in range(self.doc_size):
			doc_vec = [0.] * self.vocab_size
			for w, tf in self.title_dict[i].iteritems():
				vid = self.v2i[w]
				TFIDF = math.log(float(self.doc_size)/float(self.df[vid])) * float(tf)/float(len(self.title_data[i]))
				doc_vec[vid] = TFIDF
			self.doc_vec.append(doc_vec)

		self.doc_vec = np.array(self.doc_vec)

	def doc_vec_with_word2vec(self, word_vector):
		self.doc_vec = []
		for i in range(self.doc_size):
			doc_vec = 0.
			for w in self.title_data[i]:
				vid = self.v2i[w]
				doc_vec += word_vector[vid]
			doc_vec /= float(len(self.title_data[i]))
			self.doc_vec.append(doc_vec)

		self.doc_vec = np.array(self.doc_vec)

def build_cooccur(args, title_data, v2i, window_size, dumpAll=True):

	cooccur = []
	total = 0
	for tokens in title_data:
		total += len(tokens)
	print total
	pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=total).start()	

	index = 0
	for tokens in title_data:

		window = [-1]*window_size
		tail = 0

		for token in tokens:
			index += 1
			if token == '\n':
				tail = 0
				continue

			w2 = v2i.get(token, -1)

			current = tail - 1
			head = tail - window_size - 1 if tail > window_size else 0 - 1

			for index in range(current, head, -1):
				w1 = window[index%window_size]

				if w1 != -1 and w2 != -1:
					cooccur.append([w1, w2])
					cooccur.append([w2, w1])
					
			window[tail%window_size] = w2
			tail += 1
			
			pbar.update(index+1)
	pbar.finish()
	
	np_cooccur = None
	if len(cooccur) > 0 and dumpAll:
		np_cooccur = np.array(cooccur)
		np.random.shuffle(np_cooccur)
		np.save(args.contextword+'.npz', np_cooccur)

	print >> sys.stderr, 'done dumping cooccur file'
	return np_cooccur

def camel_case_split(identifier):
	matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
	camel = [m.group(0) for m in matches]
	out = ' '.join(camel)
	return out

def replace_char(in_str):
	in_str = in_str.replace("EXcel", 'excel')
	in_str = in_str.replace("LinQ", 'linq')
	in_str = in_str.replace("SharePoint", 'sharepoint')
	in_str = in_str.replace("WordPress", 'wordpress')
	in_str = in_str.replace("MatLab", 'matlab')
	in_str = camel_case_split(in_str)
	in_str = in_str.lower()
	in_str = in_str.replace("'m ", ' am ')
	in_str = in_str.replace("'t ", 'not ')
	in_str = in_str.replace("'s ", ' ')

	re_char = ['/', '"', '.', '?', '_', '(', ')', ',', "'", '-', ':', '<', '>', '*', '[', ']','!']

	for c in re_char:
		in_str = in_str.replace(c, ' ')

	return in_str

def get_title_data(args):
	p_stemmer = PorterStemmer()
	stop_word = ['a', 'an', 'is', 'the', 'are', 'am', 'and', 'of', 'how', 'to', 'in', 'or', 'i',
				'on', 'in', 'with','you','what','it', 'It', 'possible', 'can', 'for', 'do', 'from', 'use',
				'not', 'get', 'was', 'were','my', 'doe', 'as', 'be', 'into', 'when', 'why', 'file']
	title_data = []
	for line in open(args.title_doc, 'r'):
		line = line.strip()
		line_l = replace_char(line).split()
		tmp = []
		for i in line_l:
			try:
				tmp.append(p_stemmer.stem(i))
			except:
				tmp.append(i)
		line_l = tmp
		for s in stop_word:
			line_l = filter(lambda w : w != s, line_l)
				
		title_data.append(line_l)

	return title_data

def train(args ,vocab_size, data, batch_size, iterations, embedding_size, sample_num, learning_rate, mode):

	train_x = tf.placeholder(tf.int32, [None])
	train_y = tf.placeholder(tf.int32, [None, 1])

	with tf.device('/cpu:0'):
		uidv = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		emlt = tf.nn.embedding_lookup(uidv, train_x)
		w_nce = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		b_nce = tf.Variable(tf.zeros([vocab_size]))

	nce_loss = tf.nn.nce_loss(weights=w_nce, biases=b_nce, inputs=emlt, labels=train_y, 
				num_sampled=sample_num, num_classes=vocab_size, name="nce_loss")

	cost = tf.reduce_mean(nce_loss, name="cost")

	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

	saver = tf.train.Saver()

	with tf.Session() as sess:

		if mode == 'train':
			init = tf.initialize_all_variables()
			sess.run(init)
			# saver.restore(sess, './%s.ckpt'%args.model)

			batch_number = data.length/batch_size
			batch_number += 1 if data.length%batch_size > 0 else 0

			for ite in range(iterations):
				print >> sys.stderr, 'Iterations ',ite+1,':'
				avg_cost = 0.
				pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_number).start()	
				for b in range(batch_number):
					
					t_x, t_y = data.next_batch(batch_size)
					
					_, c = sess.run([optimizer, cost], feed_dict={train_x:t_x, train_y:t_y})
					
					avg_cost += c/batch_number

					pbar.update(b+1)
				pbar.finish()	
				print >> sys.stderr, '>>> cost : '+str(avg_cost)

			saver.save(sess, './%s.ckpt'%args.model)

		else:
			saver.restore(sess, './%s.ckpt'%args.model)

		return uidv.eval()

def predict(args, pred):
	f = open(args.pred_pairs, 'r')
	fo = open(args.pred_out, 'w')
	fo.write('ID,Ans\n')
	for i, row in enumerate(csv.reader(f)):
		if i != 0: # ignore header
			ans = 1 if pred[int(row[1])] == pred[int(row[2])] else 0
			out = str(i-1)+','+str(ans)+'\n'
			fo.write(out)

def predict2(args, doc_vec):
	f = open(args.pred_pairs, 'r')
	fo = open(args.pred_out, 'w')
	fo.write('ID,Ans\n')
	for i, row in enumerate(csv.reader(f)):
		if i != 0: # ignore header
			ans = 1 if np.dot(doc_vec[int(row[1])], doc_vec[int(row[2])]) >= 0.1 else 0
			out = str(i-1)+','+str(ans)+'\n'
			fo.write(out)

def main():

	args = arg_parse()

	title_data = get_title_data(args)

	model = Word_Struct(title_data)

	model.build_vocab()

	model.dump_vocab()
	
	train_pair = build_cooccur(args, title_data, model.v2i, window_size=5, dumpAll=True)

	data = Data(train_pair)

	print >> sys.stderr, 'done building data'

	word_vector = train(args ,model.vocab_size, data, batch_size=100, iterations=100, embedding_size=50, sample_num=100, learning_rate=0.25, mode=args.mode)

	f = open('./word_vector.txt', 'w')
	for i, v in enumerate(word_vector):
		out = str(model.i2v[i])+' '
		out += ' '.join([str(value) for value in v])+'\n'
		f.write(out)

	model.doc_vec_with_word2vec(word_vector)

	f = open('./doc_vector.txt', 'w')
	for i, v in enumerate(model.doc_vec):
		out = str(i)+' '
		out += ' '.join([str(value) for value in v])+'\n'
		f.write(out)

	print >> sys.stderr, 'start k-means training...'	
	kmeans = KMeans(n_clusters=24).fit(model.doc_vec)
	pred = kmeans.labels_

	predict(args, pred)


if __name__ == '__main__':
	main()