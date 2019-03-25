import os
import re
import sys
import json
import nltk
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def clean_str(s):
	s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

def load_embeddings(vocabulary):

	#print('Indexing word vectors.')
	GLOVE_DIR = '/home/angrosh/Documents/Angrosh/embeddings/glove.6B/'
	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	word_embeddings = {}
	for word in vocabulary:
		if word in word_embeddings:
			word_embeddings[word] = embeddings_index[word]
		else:
			word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)

	return word_embeddings

def load_dep_embeddings(vocabulary):
	word_embeddings = {}
	for word in vocabulary:
		word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
	return word_embeddings

def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=200):
	"""Pad setences during training or prediction"""
	if forced_sequence_length is None: # Train
		sequence_length = max(len(x) for x in sentences)
	else: # Prediction
		logging.critical('This is prediction, reading the trained sequence length')
		sequence_length = forced_sequence_length
	logging.critical('The maximum length is {}'.format(sequence_length))

	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)

		if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
			logging.info('This sentence has to be cut off because it is longer than trained sequence length')
			padded_sentence = sentence[0:sequence_length]
		else:
			padded_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(padded_sentence)
	return padded_sentences

def build_vocab(sentences):
	word_counts = Counter(itertools.chain(*sentences))
	vocabulary_inv = [word[0] for word in word_counts.most_common()]
	vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
	return vocabulary, vocabulary_inv

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

def load_data(filename):
	df = pd.read_csv(filename, compression='zip')
	selected = ['Category', 'Descript']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[0]].tolist())))
	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw= df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	x_raw = pad_sentences(x_raw)
	vocabulary, vocabulary_inv = build_vocab(x_raw)

	x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
	y = np.array(y_raw)

	return x, y, vocabulary, vocabulary_inv, df, labels

def get_relative_position_left(sents_list, max_sentence_length=100):
	pos_left = []
	j = 0
	while j < len(sents_list):

		sentence = sents_list[j]
		sentence = ' '.join(sentence).decode('utf-8')
		tokens = nltk.word_tokenize(sentence)

		e1 = len(tokens)-1

		d1 = ""
		for word_idx in range(len(tokens)):
			d1 += str((max_sentence_length - 1) + word_idx - e1) + " "
		for _ in range(max_sentence_length - len(tokens)):
			d1 += "999 "

		pos_left.append(d1)

		j = j+1

	return pos_left

def get_relative_position_right(sents_list, max_sentence_length=100):
	pos_right = []
	j = 0
	while j < len(sents_list):

		sentence = sents_list[j]
		sentence = ' '.join(sentence).decode('utf-8')
		tokens = nltk.word_tokenize(sentence)

		e1 = 0

		d1 = ""
		for word_idx in range(len(tokens)):
			d1 += str((max_sentence_length - 1) + word_idx - e1) + " "

		for _ in range(max_sentence_length - len(tokens)):
			d1 += "999 "

		pos_right.append(d1)

		j = j+1

	return pos_right


def get_relative_position_middle(sents_list, max_sentence_length=100):
	pos_1 = []
	pos_2 = []
	
	j = 0
	while j < len(sents_list):

		sentence = sents_list[j]
		sentence = ' '.join(sentence).decode('utf-8')
		tokens = nltk.word_tokenize(sentence)

		e1 = 0
		e2 = len(tokens)-1

		d1 = ""
		d2 = ""
		for word_idx in range(len(tokens)):
			d1 += str((max_sentence_length - 1) + word_idx - e1) + " "
			d2 += str((max_sentence_length - 1) + word_idx - e2) + " "
		for _ in range(max_sentence_length - len(tokens)):
			d1 += "999 "
			d2 += "999 "
		
		pos_1.append(d1)
		pos_2.append(d2)

		j = j+1

	return pos_1, pos_2


def get_relative_position_1(sents_list, entities_list, max_sentence_length=100):
    # Position data
    pos1 = []
    pos2 = []

    j = 0
    while j < len(sents_list):

    	sentence = sents_list[j]
    	sentence = ' '.join(sentence).decode('utf-8')
    	tokens = nltk.word_tokenize(sentence)

    	ents = entities_list[j]

    	e1 = int(ents[0])
    	e2 = int(ents[-1])

    	d1 = ""
        d2 = ""
        for word_idx in range(len(tokens)):
            d1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            d2 += str((max_sentence_length - 1) + word_idx - e2) + " "

        for _ in range(max_sentence_length - len(tokens)):
            d1 += "999 "
            d2 += "999 "

        pos1.append(d1)
        pos2.append(d2)


    	j = j+1

    return pos1, pos2


def get_relative_position(sents_list, entities_list, max_sentence_length=100):
    # Position data
    pos1 = []
    pos2 = []


    pos1_left = []
    pos2_left = []

    pos1_middle = []
    pos2_middle = []

    pos1_right = []
    pos2_right = []

    j = 0
    while j < len(sents_list):

    	sentence = sents_list[j]
    	sentence = ' '.join(sentence).decode('utf-8')
    	tokens = nltk.word_tokenize(sentence)

    	ents = entities_list[j]

    	e1 = int(ents[0])
    	e2 = int(ents[-1])

    	d1 = ""
    	d2 = ""
    	for word_idx in range(len(tokens)):
    		d1 += str((max_sentence_length - 1) + word_idx - e1) + " "
    		d2 += str((max_sentence_length - 1) + word_idx - e2) + " "

    	split_d1 = d1.split(' ')

    	left_tokens = []
    	middle_tokens = []
    	right_tokens = []

    	k = 0
    	while k < len(split_d1):
    		if k <= e1:
    			left_tokens.append(split_d1[k])
    		elif k > e1 and k < e2:
    			middle_tokens.append(split_d1[k])
    		elif k >= e2:
    			right_tokens.append(split_d1[k])
    		k = k+1

    	d1_left = ' '.join(left_tokens)
    	d1_middle = ' '.join(middle_tokens)
    	d1_right = ' '.join(right_tokens)
    	
    	for _ in range(max_sentence_length - len(split_d1)):
    		d1_left += "999 "
    		d1_middle += "999 "
    		d1_right += "999 "

    	pos1_left.append(d1_left)
    	pos1_middle.append(d1_middle)
    	pos1_right.append(d1_right)

    	split_d2 = d2.split(' ')

    	left_tokens = []
    	middle_tokens = []
    	right_tokens = []

    	k = 0
    	while k < len(split_d2):
    		if k <= e1:
    			left_tokens.append(split_d2[k])
    		elif k > e1 and k < e2:
    			middle_tokens.append(split_d2[k])
    		elif k >= e2:
    			right_tokens.append(split_d2[k])
    		k = k+1

    	d2_left = ' '.join(left_tokens)
    	d2_middle = ' '.join(middle_tokens)
    	d2_right = ' '.join(right_tokens)
    	
    	for _ in range(max_sentence_length - len(split_d2)):
    		d2_left += "999 "
    		d2_middle += "999 "
    		d2_right += "999 "

    	pos2_left.append(d2_left)
    	pos2_middle.append(d2_middle)
    	pos2_right.append(d2_right)

    	j = j+1

    return pos1_left, pos2_left, pos1_middle, pos2_middle, pos1_right, pos2_right

def load_data_split_sents(dir_path):
	
	all_sent_parts = []

	source_full_train_raw = []
	with open(dir_path+'source_full_train.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				source_full_train_raw.append(split_line)
				all_sent_parts.append(split_line)
	source_full_train_raw = pad_sentences(source_full_train_raw)


	entities_train = []
	with open(dir_path+'entities_train.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				entities_train.append(split_line)
	
	pos1_train, pos2_train = get_relative_position_1(source_full_train_raw, entities_train)


	source_full_dev_raw = []
	with open(dir_path+'source_full_dev.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				source_full_dev_raw.append(split_line)
				all_sent_parts.append(split_line)

	source_full_dev_raw = pad_sentences(source_full_dev_raw)


	entities_dev = []
	with open(dir_path+'entities_dev.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				entities_dev.append(split_line)

	pos1_dev, pos2_dev = get_relative_position_1(source_full_dev_raw, entities_dev)

	source_full_test_raw = []
	with open(dir_path+'source_full_test.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				source_full_test_raw.append(split_line)
				all_sent_parts.append(split_line)

	source_full_test_raw = pad_sentences(source_full_test_raw)

	entities_test = []
	with open(dir_path+'entities_test.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				entities_test.append(split_line)

	pos1_test, pos2_test = get_relative_position_1(source_full_test_raw, entities_test)

	all_sent_parts = pad_sentences(all_sent_parts)

	vocabulary, vocabulary_inv = build_vocab(all_sent_parts)
	
	x_train = np.array([[vocabulary[word] for word in sentence] for sentence in source_full_train_raw])
	x_dev = np.array([[vocabulary[word] for word in sentence] for sentence in source_full_dev_raw])
	x_test = np.array([[vocabulary[word] for word in sentence] for sentence in source_full_test_raw])

	y_raw_tmp = []
	with open(dir_path+'relations_train.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				y_raw_tmp.append(line)

	with open(dir_path+'relations_dev.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				y_raw_tmp.append(line)

	with open(dir_path+'relations_test.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				y_raw_tmp.append(line)
	labels = sorted(list(set(y_raw_tmp)))

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_raw_tmp = []
	with open(dir_path+'target_train.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				tmp = []
				for term in split_line:
					tmp.append(int(term))
				y_raw_tmp.append(tmp)
	y_raw = np.asarray(y_raw_tmp)
	y_train = np.array(y_raw_tmp)

	y_raw_tmp = []
	with open(dir_path+'target_dev.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				tmp = []
				for term in split_line:
					tmp.append(int(term))
				y_raw_tmp.append(tmp)
	y_raw = np.asarray(y_raw_tmp)
	y_dev = np.array(y_raw_tmp)

	y_raw_tmp = []
	with open(dir_path+'target_test.txt', 'r') as f:
		for line in f:
			if line:
				line = re.sub('\n', '', line)
				split_line = line.split(' ')
				tmp = []
				for term in split_line:
					tmp.append(int(term))
				y_raw_tmp.append(tmp)
	y_raw = np.asarray(y_raw_tmp)
	y_test = np.array(y_raw_tmp)

	return x_train, x_dev, x_test, pos1_train, pos2_train, pos1_dev, pos2_dev, pos1_test, pos2_test, y_train, y_dev, y_test, vocabulary, vocabulary_inv, labels




if __name__ == "__main__":
	#train_file = './data/train.csv.zip'
	#load_data(train_file)

	dir_path = '../data/drug_var_single_sent_binary_relation_ents_dep_nodes_10_fold/fold_0/'
	load_data_split_sents(dir_path)
