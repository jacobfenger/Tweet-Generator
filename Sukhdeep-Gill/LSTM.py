print("starting of the file")
import tensorflow as tf
print(tf.test.gpu_device_name())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import pandas as pd
import numpy as np
import re
import sys, os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SpatialDropout1D, Bidirectional, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys
import io
print("imports done")


unique_chars2=[]  
chars2 = []
token_dict_char_int2={}
token_dict_int_char2={}
maxlen2=0
model2 = ""

def readInputFile(fileName):
    df = pd.read_csv(fileName, encoding = "ISO-8859-1", header=None).iloc[:, 5]
    
    return df
    
   
# process a single tweet
def preprocess(tweet):
    
    # lowercase all the tweets
    tweet = tweet.lower()
    
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


def getChars(preprocessed_tweets):

    # change the input data into one big string and
    # count the total number of characters 
    chars = ' '.join(preprocessed_tweets).lower()
    
    # get the unique characters from the text
    unique_chars = sorted(list(set(' '.join(preprocessed_tweets).lower())))
    
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
    
    
    
    
    
def getInput(chars):
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(chars) - maxlen, step):
        sentences.append(chars[i: i + maxlen])
        next_chars.append(chars[i + maxlen])


    print('Number of sequences:', len(sentences), "\n")
    print(sentences[:10], "\n")
    print(next_chars[:10])
    return maxlen, sentences, next_chars
    
    
def InOut(unique_chars, chars, token_dict_char_int):
    maxlen, sentences, next_chars = getInput(chars)
    sentences = sentences[:100000]
    x = np.zeros((len(sentences), maxlen, len(unique_chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(unique_chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, token_dict_char_int[char]] = 1
        y[i, token_dict_char_int[next_chars[i]]] = 1
        
    return x, y, maxlen
        
        
        
        
def getModel(maxlen, unique_chars):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(unique_chars))))
    model.add(Dense(len(unique_chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
    
    
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
    
    
def on_epoch_end(epoch, logs):
    maxlen = 40
    print("Hello")
    global unique_chars2
    global chars2
    global token_dict_char_int2
    global token_dict_int_char2
    global maxlen2
    global model2
    # Function invoked for specified epochs. Prints generated text.
    # Using epoch+1 to be consistent with the training epochs printed by Keras
    if epoch+1 == 1 or epoch+1 == 15:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        print(maxlen2)
        print(len(chars2))
        print(len(unique_chars2))
        start_index = random.randint(0, len(chars2) - maxlen2 - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = chars2[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen2, len(unique_chars2)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, token_dict_char_int2[char]] = 1.

                preds = model2.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = token_dict_int_char2[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)




def fitModel(maxlen, unique_chars, token_dict_char_int, token_dict_int_char, chars, x, y):
    filepath = "weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')
    
    model = getModel(maxlen, unique_chars)
    global model2
    model2 = model
    generate_text = LambdaCallback(on_epoch_end=on_epoch_end)
    
    with tf.device('/gpu:0'):
        model.fit(x, y,
                  batch_size=128,
                  epochs=15,
                  verbose=2,
                  callbacks=[generate_text, checkpoint])
              #callbacks=[generate_text, checkpoint])
    
    model.save("LSTM_new_GPU.h5")
    return model
   
def main():
    df = readInputFile("training_data.csv")
    preprocessed_tweets = df.apply(preprocess).values 
    chars1, unique_chars1, no_chars1, no_unique_chars1, token_dict_char_int1, token_dict_int_char1 = getChars(preprocessed_tweets)
    x, y, maxlen1 = InOut(unique_chars1, chars1, token_dict_char_int1)
    global unique_chars2
    unique_chars2=unique_chars1
    global chars2
    chars2=chars1
    global token_dict_char_int2
    token_dict_char_int2=token_dict_char_int1
    global token_dict_int_char2
    token_dict_int_char2=token_dict_int_char1
    global maxlen2 
    maxlen2= maxlen1
    print(maxlen2)
    print(len(chars2))
    print(len(unique_chars2))
    fitModel(maxlen1, unique_chars1, token_dict_char_int1, token_dict_int_char1, chars1, x, y)
    
    
    
    
if __name__ == '__main__':
    main()