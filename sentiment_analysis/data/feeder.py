import numpy as np 
import pandas as pd 
import os
import threading
import time
import traceback
from utils.infolog import log
import tensorflow as tf 
from hparams import hparams


_batches_per_group = 32
_pad = 0

class Feeder(threading.Thread):
	"""
		Feeds batches of data into queue on a background thread
	"""

	def __init__(self, coordinator, metadata_filename, hparams):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._offset = 0

		#Load metadata
		self._datadir = os.path.dirname(metadata_filename)
		self._metadata = pd.read_csv(metadata_filename, delimiter='\t', names=['label', 'sequence'])

		# Create placeholders for inputs and targets. Don't specify batch size because we want
		# to be able to feed different batch sizes at eval time.
		self._placeholders = [
		tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
		tf.placeholder(tf.float32, shape=(None, self._hparams.num_classes), name='targets')]

		#Create queue for buffering data
		queue = tf.FIFOQueue(8, [tf.int32, tf.float32], name='input_queue')
		self._enqueue_op = queue.enqueue(self._placeholders)
		self.inputs, self.targets = queue.dequeue()
		self.inputs.set_shape(self._placeholders[0].shape)
		self.targets.set_shape(self._placeholders[1].shape)

	def start_in_session(self, session):
		self._session = session
		self.start()

	def run(self):
		try:
			while not self._coord.should_stop():
				self.enqueue_next_group()
		except Exception as e:
			traceback.print_exc()
			self._coord.request_stop(e)

	def _enqueue_next_group(self):
		start = time.time()

		#Read a group of samples
		n = self._hparams.batch_size
		examples = [self._get_next_example() for i in range(n * _batches_per_group)]

		batches = [examples[i: i+n] for i in range(0, len(examples), n)]
		np.random.shuffle(batches)

		log('\nGenerated {} batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		for batch in batches:
			feed_dict = dict(zip(self._placeholders, _prepare_batch(batch)))
			self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		"""
			Gets a simple example (input, target) for disk
		"""
		if self._offset >= len(self._metadata):
			self._offset = 0
			np.random.shuffle(self._metadata)

		meta = self._metadata[self._offset]
		self._offset += 1

		text = meta[1]

		input_data = np.asarray(text, dtype=np.int32)
		target = meta[0].astype(np.float32)
		return (input_data, target)

def _prepare_batch(batch):
	np.random.shuffle(batch)
	inputs = _prepare_inputs([x[0] for x in batch])
	targets = _prepare_targets([x[1] for x in batch])
	return (inputs, targets)

def _prepare_inputs(inputs):
	max_len = hparams.max_len
	return np.stack([_pad_input(x, max_len) for x in inputs])

def _prepare_targets(targets):
	return targets.astype(np.float32)

def _pad_input(x, length):
	return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)
