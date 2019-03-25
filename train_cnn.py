import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn import TextCNN
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

def train_cnn():
	input_dir = sys.argv[1]
	
	x_train, x_dev, x_test, pos1_train, pos2_train,pos1_dev, pos2_dev, pos1_test, pos2_test, y_train, y_dev, y_test, vocabulary, vocabulary_inv, labels  = data_helper.load_data_split_sents(input_dir)

	training_config = sys.argv[2]
	params = json.loads(open(training_config).read())


	# Assign a 300 dimension vector to each word
	word_embeddings = data_helper.load_embeddings(vocabulary)
	embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
	embedding_mat = np.array(embedding_mat, dtype = np.float32)

	#sentence_length = 200
	pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(200)
	pos_vocab_processor.fit(pos1_train + pos2_train + pos1_dev + pos2_dev + pos1_test + pos2_test)

	pos1_train_vec = np.array(list(pos_vocab_processor.transform(pos1_train)))
	pos2_train_vec = np.array(list(pos_vocab_processor.transform(pos2_train)))
	
	pos1_dev_vec = np.array(list(pos_vocab_processor.transform(pos1_dev)))
	pos2_dev_vec = np.array(list(pos_vocab_processor.transform(pos2_dev)))

	pos1_test_vec = np.array(list(pos_vocab_processor.transform(pos1_test)))
	pos2_test_vec = np.array(list(pos_vocab_processor.transform(pos2_test)))
	


	# Create a directory, everything related to the training will be saved in this directory
	timestamp = str(int(time.time()))
	trained_dir = './trained_results_' + timestamp + '/'
	if os.path.exists(trained_dir):
		shutil.rmtree(trained_dir)
	os.makedirs(trained_dir)

	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
				embedding_mat=embedding_mat,
				sequence_length=x_train.shape[1],
				num_classes = y_train.shape[1],
				non_static=params['non_static'],
				hidden_unit=params['hidden_unit'],
				max_pool_size=params['max_pool_size'],
				filter_sizes=map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				embedding_size = params['embedding_dim'],
				pos_vocab_size=len(pos_vocab_processor.vocabulary_),
                pos_embedding_size=params['position_embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			global_step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Checkpoint files will be saved in this directory during training
			checkpoint_dir = './checkpoints_' + timestamp + '/'
			if os.path.exists(checkpoint_dir):
				shutil.rmtree(checkpoint_dir)
			os.makedirs(checkpoint_dir)
			checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def train_step(x1_batch, pos1_batch, pos2_batch, y_batch):
				feed_dict = {
					cnn.input_x1: x1_batch,
					cnn.input_pos1: pos1_batch,
					cnn.input_pos2: pos2_batch,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: params['dropout_keep_prob'],
					cnn.batch_size: len(x1_batch),
					cnn.pad: np.zeros([len(x1_batch), 1, params['embedding_dim'], 1]),
					#cnn.pad_pos: np.zeros([len(x1_batch), 1, params['embedding_dim']+2*params['position_embedding_dim'], 1]),
					cnn.real_len: real_len(x1_batch),
				}
				_, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

			def dev_step(x1_batch, pos1_batch, pos2_batch, y_batch):
				feed_dict = {
					cnn.input_x1: x1_batch,
					cnn.input_pos1: pos1_batch,
					cnn.input_pos2: pos2_batch,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: 1.0,
					cnn.batch_size: len(x1_batch),
					cnn.pad: np.zeros([len(x1_batch), 1, params['embedding_dim'], 1]),
					#cnn.pad_pos: np.zeros([len(x1_batch), 1, params['embedding_dim']+2*params['position_embedding_dim'], 1]),
					cnn.real_len: real_len(x1_batch),
				}
				step, loss, accuracy, num_correct, predictions = sess.run(
					[global_step, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.predictions], feed_dict)
				return accuracy, loss, num_correct, predictions

			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())

			# Training starts here
			train_batches = data_helper.batch_iter(list(zip(x_train, pos1_train_vec, pos2_train_vec, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0

			# Train the model with x_left_train and y_train
			print "len(train_batches): ", train_batches
			prev_test_set_accuracy = 0.0
			for train_batch in train_batches:
				#print train_batch
				if train_batch.shape[0] > 0:

					x_train_batch, pos1_train_batch, pos2_train_batch, y_train_batch = zip(*train_batch)

					train_step(x_train_batch, pos1_train_batch, pos2_train_batch, y_train_batch)
					current_step = tf.train.global_step(sess, global_step)


					# Evaluate the model with x_left_dev and y_dev
					if current_step % params['evaluate_every'] == 0:
						dev_batches = data_helper.batch_iter(list(zip(x_dev, pos1_dev_vec, pos2_dev_vec, y_dev)), params['batch_size'], 1)

						total_dev_correct = 0
						count_y_dev = 0
						for dev_batch in dev_batches:
							if dev_batch.shape[0] > 0:
								x_dev_batch, pos1_dev_batch, pos2_dev_batch, y_dev_batch = zip(*dev_batch)
								acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, pos1_dev_batch, pos2_dev_batch, y_dev_batch)
								total_dev_correct += num_dev_correct
								count_y_dev = count_y_dev + len(dev_batch)
						accuracy = float(total_dev_correct) / count_y_dev
						logging.info('Accuracy on dev set: {}'.format(accuracy))
	

						test_batches = data_helper.batch_iter(list(zip(x_test, pos1_test_vec, pos2_test_vec, y_test)), params['batch_size'], 1, shuffle=False)
						total_test_correct = 0
						count_y_test = 0
						for test_batch in test_batches:
							if test_batch.shape[0] > 0:
								x_test_batch, pos1_test_batch, pos2_test_batch, y_test_batch = zip(*test_batch)
								acc, loss, num_test_correct, predictions = dev_step(x_test_batch, pos1_test_batch, pos2_test_batch, y_test_batch)
								total_test_correct += int(num_test_correct)
								count_y_test = count_y_test + len(test_batch)

						test_set_acc = float(total_test_correct) / count_y_test
						logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / count_y_test))

						if test_set_acc > prev_test_set_accuracy:
							prev_test_set_accuracy = test_set_acc
							best_accuracy, best_at_step = accuracy, current_step
							path = saver.save(sess, checkpoint_prefix, global_step=current_step)
							logging.critical('Saved model {} at step {}'.format(path, best_at_step))
							logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
							logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / count_y_test))

			logging.critical('Training is complete, testing the best model on x_left_test and y_test')

			
			# Save the model files to trained_dir. predict.py needs trained model files. 
			saver.save(sess, trained_dir + "best_model.ckpt")

			# Evaluate x_left_test and y_test
			saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
			test_batches = data_helper.batch_iter(list(zip(x_test, pos1_test_vec, pos2_test_vec, y_test)), params['batch_size'], 1, shuffle=False)
			total_test_correct = 0
			count_y_test = 0
			for test_batch in test_batches:
				if test_batch.shape[0] > 0:
					x_test_batch, pos1_test_batch, pos2_test_batch, y_test_batch = zip(*test_batch)
					acc, loss, num_test_correct, predictions = dev_step(x_test_batch, pos1_test_batch, pos2_test_batch, y_test_batch)
					total_test_correct += int(num_test_correct)
					count_y_test = count_y_test + len(test_batch)
			logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / count_y_test))



	# Save trained parameters and files since predict.py needs them
	with open(trained_dir + 'words_index.json', 'w') as outfile:
		json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
	with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
		pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
	with open(trained_dir + 'labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4, ensure_ascii=False)

	params['sequence_length'] = x_train.shape[1]
	with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
	train_cnn()
