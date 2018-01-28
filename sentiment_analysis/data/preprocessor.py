import pandas as pd 
import os
import argparse
import nltk
from nltk.tokenize import TweetTokenizer
import csv


_eos = -1

def read_data(file, training):
	if training:
		data = pd.read_csv(file, delimiter='\t', names=['label', 'text'], header=None)
	else:
		data = pd.read_csv(file, delimiter='\t', names=['text'], header=None)
	
	return data

def create_vocabulary(texts, vocab_file):
	tknzr=TweetTokenizer()
	word_vocab = {}
	def func(text):
		#for word in nltk.word_tokenize(text):
		for word in tknzr.tokenize(text):
			if word not in word_vocab:
				word_vocab[word] = 1
			else:
				word_vocab[word] += 1
		return 0

	_ = texts.apply(func)

	with open(vocab_file, 'w') as file:
		for key, value in sorted(word_vocab.items(), key=lambda x: (x[1], x[0]), reverse=True):
			file.write('{}|{}\n'.format(value, key))

	print('created vocabulary at {}'.format(vocab_file))

def text_to_sequence(text, vocab_file):
	"""Converts a string to a sequence of IDs corresponding to words in vocabulary
	"""
	vocab = pd.read_csv(vocab_file, delimiter='|', names=['count', 'word'], header=None, quoting=csv.QUOTE_NONE)
	#convert string to list of strings
	tknzr=TweetTokenizer()
	sequence = tknzr.tokenize(text)

	#convert strings to their IDs
	sequence = list(map(lambda x: vocab.index[vocab['word'] == x][0] + 1, sequence))

	#add an eos token
	sequence.append(_eos)
	return sequence

def sequence_to_text(sequence, vocab_file):
	"""Converts a sequence of IDs corresponding to words to a string
	"""
	vocab = pd.read_csv(vocab_file, delimiter='|', names=['count', 'word'], quoting=csv.QUOTE_NONE)
	#convert IDs to words
	sequence = map(lambda x: vocab['word'][x - 1], sequence)

	#invert tokenize sequence to string
	text = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sequence]).strip()
	return text

def class_to_str(cl):
	"""Converts the label of a class to the corresponding name
	"""
	return "positive" if cl == 1 else "negative"

def preprocess(in_file, out_file, vocab_file, training):
	data = read_data(in_file, training=training)
	texts = data['text']
	if training:
		test_data = read_data('../data/test.csv', training=False)
		texts = pd.concat([texts, test_data['text']], axis=0)
		create_vocabulary(texts, vocab_file)
	
	print('creating output file: {}'.format(out_file))

	with open(out_file, 'w') as file:
		for i in range(len(data.index)):
			if training:
				file.write('{}\t{}\n'.format(data['label'][i], text_to_sequence(data['text'][i], vocab_file)))
			else:
				file.write('{}\n'.format(text_to_sequence(data['text'][i], vocab_file)))
	print('preprocessing complete')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default=os.path.dirname(os.path.realpath(__file__)))
	parser.add_argument('--input', default='train.csv')
	parser.add_argument('--output_dir', default='../datasets')
	parser.add_argument('--training', type=bool, default=False, help='Set this to False to preprocess text set')
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	output_dir = args.output_dir

	if args.training:
		input_file = 'train.csv'
		output_file = os.path.join(output_dir, 'train.csv')
	else:
		input_file = 'test.csv'
		output_file = os.path.join(output_dir, 'test.csv')

	vocab_file = os.path.join(output_dir, 'vocab_file.csv')

	preprocess(input_file, output_file, vocab_file, args.training)


if __name__ == '__main__':
	main()