import tensorflow as tf 
from utils.symbols import symbols
from utils.infolog import log
from .modules import *


class Analyser():
	def __init__(self, hparams):
		self._hparams = hparams

	def initialize(self, inputs, targets=None):
		"""
		Initializes the model for inference

		set "output" field.

		Args:
			- inputs: int32 tensor with shape [batch_size, time_steps] where time steps
			is typically the number of words in each input sentence
			- targets: int32 tensor with shape [batch_size, num_classes] which represents the true labels.
			Only used in training time. 
		"""
		with tf.variable_scope('inference') as scope:
			is_training = targets is not None
			batch_size = tf.shape(inputs)[0]
			hp = self._hparams

			#Embeddings
			embedding_table = tf.get_variable(
				'intputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32,
				initializer=tf.truncated_normal_initializer(stddev=0.5))
			embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)

			#Encoder
			enc_conv_outputs = enc_conv_layers(embedded_inputs, is_training)        
			encoder_outputs, encoder_states = bidirectional_LSTM(enc_conv_outputs, 
				'encoder_LSTM', is_training=is_training) 

			#Prediction/projection
			projection_shape = [512, 512]
			projected = projection_layers(inputs, is_training=is_training,
										  shape=projection_shape,
										  activation=tf.nn.relu)

			#Logit Layer
			output = logit_layer(projected, logits_dim=hp.num_classes)

			self.inputs = inputs
			self.output = output
			self.targets = targetst
			og('Initialized Analyser model. Dimensions: ')
			log('  embedding:               {}'.format(embedded_inputs.shape[-1]))
			log('  enc conv out:            {}'.format(enc_conv_outputs.shape[-1]))
			log('  encoder out:             {}'.format(encoder_outputs.shape[-1]))
			log('  output:                  {}'.format(output.shape[-1]))


	def add_loss(self):
		"""
		Adds loss to model, sets "loss" field, initialize must have been called
		"""
		with tf.variable_scope('loss') as scope:
			hp = self._hparams

			#Compute network loss
			net_loss = tf.losses.softmax_cross_entropy(self.targets, self.output)

			#Add regularization term
			all_vars = tf.trainable_variables()
			regularization = tf.add_n([tf.nn.l2.loss(v) for v in all_vars]) * hp.reg_weight

			#Compute total loss
			self.net_loss = net_loss
			self.regularization_loss = regularization
			self.loss = self.net_loss + self.regularization_loss

	def add_optimizer(self, global_step):
		"""
		Adds optimizer. Sets "gradients"  and "optimize" fields. add_loss must have been called.

		Ags:
			global_step: int32 scalar tensor representing current global step in training.
		"""
		with tf.variable_scope('optimizer') as scope:
			hp = self._hparams
			self.learning_rate = tf.convert_to_tensor(hp.learning_rate)

			self.optimize = tf.train.AdamOptimizer(self.learning_rate,
												   hp.adam_beta1,
												   hp.adam_beta2,
												   hp.adam_epsilon).minimize(self.loss,
												   							 global_step=global_step)
			