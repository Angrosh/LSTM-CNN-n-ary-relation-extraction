import numpy as np
import tensorflow as tf

class TextLSTMCNN(object):
	def __init__(self, embedding_mat, non_static, hidden_unit, sequence_length, max_pool_size,
		num_classes, embedding_size, pos_vocab_size, pos_embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name='input_x1')

		self.input_pos1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos1')
		self.input_pos2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos2')

		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')

		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
		self.batch_size = tf.placeholder(tf.int32, [])
		self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
		self.pad_pos = tf.placeholder(tf.float32, [None, 1, embedding_size+2*pos_embedding_size, 1], name='pad')
		
		self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

		l2_loss = tf.constant(0.0)

		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			if not non_static:
				W = tf.constant(embedding_mat, name='W')
			else:
				W = tf.Variable(embedding_mat, name='W')
			

			self.embedded_chars_1 = tf.nn.embedding_lookup(W, self.input_x1)
			self.emb_1 = tf.expand_dims(self.embedded_chars_1, -1)


		with tf.device('/cpu:0'), tf.name_scope("position-embedding"):
			self.W_position = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embedding_size], -1.0, 1.0), name="W_position")
			
			self.pos1_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos1)
			self.pos1_embedded_chars_expanded = tf.expand_dims(self.pos1_embedded_chars, -1)
			
			self.pos2_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos2)
			self.pos2_embedded_chars_expanded = tf.expand_dims(self.pos2_embedded_chars, -1)


		#self.emb_1_expanded = tf.concat([self.emb_1, self.pos1_embedded_chars_expanded,
		#								 self.pos2_embedded_chars_expanded], 2)
		
		self.emb_1_expanded = tf.concat([self.emb_1, self.pos1_embedded_chars_expanded,
										 self.pos2_embedded_chars_expanded], 2)

		mod_emd_size_1 = embedding_size + 2*pos_embedding_size

		#2. LSTM LAYER ######################################################################
		self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_unit, state_is_tuple=True)

		#self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
		self.lstm_out,self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, self.embedded_chars_1, dtype=tf.float32)
		#embed()

		self.lstm_out_expanded = tf.expand_dims(self.lstm_out, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.lstm_out_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")

				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		num_filters_total = num_filters * len(filter_sizes)

		self.h_pool = tf.concat(pooled_outputs, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		# Calculate mean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		with tf.name_scope('num_correct'):
			correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))