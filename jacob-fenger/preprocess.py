import numpy as np
import pandas as pd
import os 
import sys
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

token_dict_int_char = None
def preprocess_tweet(tweet):
	tweet.lower()   
	# remove urls
	tweet = re.sub('((www\.\w+\.\w+) | (https?://\w+\.\w+))', '', tweet)
	#remove emails
	tweet = re.sub('(\w+)\s*(?:@|&#x40\.|\s+[aA][tT]\s+|\s*\(\s*[aA][tT]\s*\)\s*)\s*([\w\s\.]+)\s*\.\s*([eE][dD][uU]|[cC][oO][mM]|[gG][oO][vV]|[oO][rR][gG])', '', tweet)
	# remove hashtag from the front of the topic
	tweet = re.sub('#(\w+)', r'\1', tweet)
	# remove @users
	tweet = re.sub('\s*@\w+\s*', '', tweet)
	# remove multiple spaces with only one space
	tweet = re.sub('\s+', ' ', tweet)
	
	return tweet 

def tokenize_tweets(tweets):
	# Max number of words in vocab
	max_features = 450000
	maxlen = 70 # Max length for tweets
	tokenizer = Tokenizer(num_words=max_features) 

	# Fit on the tweets vocab
	tokenizer.fit_on_texts(tweets)
	# Convert from string to tokens
	tokenized_tweets = tokenizer.texts_to_sequences(tweets)
	# Pad each tweet vector to ensure they are all the same length
	tweets_train = pad_sequences(tokenized_tweets, maxlen=maxlen)

	return tweets_train

def getChars(preprocessed_tweets):

    # change the input data into one big string and
    # count the total number of characters 
    chars = ' '.join(preprocessed_tweets).lower()
    
    # get the unique characters from the text
    unique_chars = sorted(list(set(' '.join(preprocessed_tweets).lower())))
    unique_chars[:10]
    
    no_chars = len(chars)
    no_unique_chars = len(unique_chars)
    print("Total number of characters: ", no_chars)
    print("Total number of unique characters: ", no_unique_chars)
    
    
    # Assign a token to each character
    token_dict_char_int = {}
    token_dict_int_char = {}
    for index, char in enumerate(unique_chars):
        token_dict_char_int[char] = index    
        token_dict_int_char[index] = char
    
    return chars, unique_chars, no_chars, no_unique_chars, token_dict_char_int, token_dict_int_char

def preprocess_data():
	df = pd.read_csv("../training_data.csv", encoding = "ISO-8859-1", header=None).iloc[:, 5]
	print(df.head())

	preprocessed_tweets = df.apply(preprocess_tweet).values
	#tweets = tokenize_tweets(preprocessed_tweets)


	chars, unique_chars, no_chars, no_unique_chars, \
		   token_dict, token_dict_int_char = getChars(preprocessed_tweets)

	window_size = 100
	no_chars = 200000
	chars = chars[:200000]

	train_x = []
	train_y = []

	for i in range(0, no_chars - window_size, 1):
		x = chars[i:i+window_size]
		y = chars[i+window_size]

		input_sequence = []
		for character in x:
			input_sequence.append(token_dict[character])

		train_x.append(input_sequence)
		train_y.append(token_dict[y])

	# reshape training data (samples, time steps, features)
	x = np.reshape(train_x, (len(train_x), window_size, 1))

	# normalize the training data
	x = x/float(no_unique_chars)

	# tranform the output using one hot encoding
	y = np_utils.to_categorical(train_y)

	return train_x, train_y, window_size, x, y, no_unique_chars, token_dict_int_char

def getModel(x, y):
	file_name = 'weights.hdf5'

	model = Sequential()

	# add LSTM layer
	model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
	model.add(Dropout(0.3))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	# record all of the network weights each time loss is improved at the end of the epoch
	checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose='1', save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	return model, callbacks_list, file_name

def generateText(file_name, model, train_x, sequence_length, no_unique_chars, token_dict_int_char):
	model.load_weights(file_name)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	rand_start = np.random.randint(0, len(train_x) - 1)
	print(rand_start)
	starting_seq = train_x[rand_start]

	#print(starting_seq)

	# print("Seed: ")
	# print("\"", ''.join([token_dict_int_char[val] for val in starting_seq]))

	# generate characters
	for i in range(sequence_length):
		x = np.reshape(starting_seq, (1, len(starting_seq), 1))
		x = x/float(no_unique_chars)

		pred = model.predict(x, verbose=0)
		index = np.argmax(pred)
		final_output = token_dict_int_char[index]
		x_in = [token_dict_int_char[val] for val in starting_seq]
		#sys.stdout.write(final_output)
		starting_seq.append(index)
		starting_seq = starting_seq[1:len(starting_seq)]

	final_tweet = ''.join([token_dict_int_char[val] for val in starting_seq])

	print('\n', final_tweet) 
  
def fitModel(x, y):
	# Get model object
	model, callbacks_list, file_name = getModel(x, y)

	# Fit the model with training data
	#model.fit(x, y, epochs=20, batch_size=128, callbacks=callbacks_list)

	return model, file_name

def main():
	train_x, train_y, window_size, x, y, no_unique_chars, token_dict_int_char = preprocess_data()

	print(len(train_x))
		
	fitted_model, file_name = fitModel(x, y)
	file_name = 'weights.hdf5'
	
	for i in range(50):
		generateText(file_name, fitted_model, train_x, 155, no_unique_chars, token_dict_int_char)

if __name__ == '__main__':
	main()
