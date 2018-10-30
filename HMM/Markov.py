# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import os
import re
import sys
import getopt
import time
import pprint
import csv
	
class Markov:
  def __init__(self):
    self.states = {} #map vocab to index in counts and transitions
    self.counts  = [] #1D array
    self.transitions = [] #2D array of probabilities
    self.num_states = 0
    self.vocab = []
	
  def recoverChar(self, index):
    for w in self.vocab:
      if(self.states[w] == index):
        return w
    return None
  

# From Sukhdeep's Preprocess_Data.ipynb
# process a single tweet
def preprocess(tweet):
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
  
def getCharVocab(filename):
  vocab = ["start", "end"]
  ifs = open(filename, "r")
  lines = ifs.read()
  lines = lines.split("\n")
  
  for line in lines:
    for char in line:
      if char not in vocab and char != '':
        vocab += [char]
		
  return vocab
  
def getWordVocab(filename):
  vocab = ["_START_", "_END_"] #assumption that no tweet has these words
  ifs = open(filename, "r")
  lines = ifs.read()
  lines = lines.split("\n")
  
  for line in lines:
    line = line.split(" ")
    for ele in line:
      if ele not in vocab and ele != '':
        vocab += [ele]
  
  return vocab
  
  
def buildCharMarkov(markov, filename):
  #Initialize Markov
  markov.vocab = getCharVocab(filename)
  for char in markov.vocab:
    markov.states[char] = markov.num_states
    markov.num_states += 1
  markov.counts = np.zeros(markov.num_states)
  markov.transitions = np.zeros((markov.num_states, markov.num_states))
  
  # Get counts of occurances and transitions
  ifs = open(filename, "r")
  lines = ifs.read()
  lines = lines.split("\n")
  ifs.close()
  
  for line in lines:
    prev = markov.states["start"] #index of "start" symbol
    markov.counts[prev] += 1
    for char in line:	
      curr = markov.states[char] # index of current symbol
      markov.counts[curr] += 1
      markov.transitions[prev][curr] += 1
      prev = curr
    curr = markov.states["end"] #index of "end" symbol
    markov.counts[curr] += 1
    markov.transitions[prev][curr] += 1
	
  for i in range(len(markov.transitions)):
    markov.transitions[i] /= markov.counts[i]

  return markov
  
  
def buildWordMarkov(markov, filename):
  #Initialize Markov
  markov.vocab = getWordVocab(filename)
  for char in markov.vocab:
    markov.states[char] = markov.num_states
    markov.num_states += 1
  markov.counts = np.zeros(markov.num_states)
  markov.transitions = np.zeros((markov.num_states, markov.num_states))
  
  # Get counts of occurances and transitions
  ifs = open(filename, "r")
  lines = ifs.read()
  lines = lines.split("\n")
  ifs.close()
  
  for line in lines:
    if line != "":
      prev = markov.states["_START_"] #index of "start" symbol
      markov.counts[prev] += 1
      line = line.split(" ")
      for ele in line:
        if ele != "":
          curr = markov.states[ele] # index of current symbol
          markov.counts[curr] += 1
          markov.transitions[prev][curr] += 1
          prev = curr
      curr = markov.states["_END_"] #index of "end" symbol
      markov.counts[curr] += 1
      markov.transitions[prev][curr] += 1
	
  # +1 Smoothing
  markov.transitions += 1
  for i in range(len(markov.transitions)):
    markov.transitions[i] /= (markov.counts[i] + markov.num_states)

  return markov

def loadCharMarkov(markov, filename):
  ifs = open(filename, "r")
  lines = ifs.read()
  lines = lines.split("\n")
  ifs.close()
  
  #Ignore empty lines
  if(lines[len(lines)-1] == ""):
    lines.pop(len(lines)-1)
  
  #Initialize Markov
  # Get vocab from first line
  vocab = lines[0].split(", ")
  print("vocab size = ", len(vocab))
  for ele in vocab:
    markov.vocab.append(ele)
  
  for w in markov.vocab:
    markov.states[w] = markov.num_states
    markov.num_states += 1
  print("num states = ", markov.num_states)
  
  # Fill in transition probabilities
  markov.transitions = np.zeros((markov.num_states, markov.num_states))
  for i in range(1, len(lines)):
    markov.transitions[i-1] = lines[i].split(", ")

  return markov
  
def loadWordMarkov(markov, filename):
  ifs = open(filename, "r")
  line = ifs.readline()
  
  vocab = line.split(", ")
  for w in vocab:
    markov.vocab.append(w)
    markov.states[w] = markov.num_states
    markov.num_states += 1
  
  markov.transitions = [[] for x in range(markov.num_states)]
  i = 0
  for line in ifs:
    if line != "": #ignore empty lines
      markov.transitions[i] = line.split(", ")
      i += 1
  
  ifs.close()
  return markov
  
def generate_tweet(start_char, length, markov):
  tweet = [start_char]
  i = 0
  char = start_char
  next_state = np.arange(markov.num_states)
  while i != length:
    curr = markov.states[char] # index of current char
    next = np.random.choice(next_state,replace=True,p=markov.transitions[curr])
    tweet.append(markov.recoverChar(next))
    i += 1
  return tweet

def main():
  data_file = "training.1600000.processed.noemoticon.csv"
  raw_tweet_file = "raw_tweet_text.txt"
  #processed_file = "processed_tweet_text.txt"
  processed_file = "processed10000.txt"
  markov_file = "markov_word_processed.csv"
  
  time_start = time.time()
  
  # Read Data from data_file
  (options, args) = getopt.getopt(sys.argv[1:], '')
  if len(args) == 1 and args[0] == "data":
    ifs = open(data_file, "r")
    ofs1 = open(tweet_file, "w")
    ofs2 = open(processed_file, "w")
    raw_lines = ifs.read()
    raw_lines = raw_lines.split("\n")
    for line in raw_lines:
      line = line.strip("\"")
      line = line.split("\",\"")
      tweet = line[len(line)-1]
      processed_tweet = preprocess(tweet)
      ofs1.write(tweet + "\n")
      ofs2.write(processed_tweet + "\n")
    ofs1.close()
    ofs2.close()
  
  markov = Markov()
  # Build Char Markov based on default file
  if len(args) == 1 and args[0] == "char": #build char Markov
    markov = buildCharMarkov(markov, processed_file)
    ofs = open(markov_file, "w")
    for i in range(len(markov.vocab)-1):
      ofs.write(markov.vocab[i] + ", ")
    ofs.write(markov.vocab[len(markov.vocab)-1] + "\n")
    for row in markov.transitions:
      ofs.write(str(row[0]))
      for i in range(1, len(row)):
        ofs.write(", " + str(row[i]))
      ofs.write("\n")
    ofs.close()

  # Build Word Markov based on default file
  if len(args) == 1 and args[0] == "word": #build word Markov (default)
    markov = buildWordMarkov(markov, processed_file)
    ofs = open(markov_file, "w")
    for i in range(len(markov.vocab)-1):
      ofs.write(markov.vocab[i] + ", ")
    ofs.write(markov.vocab[len(markov.vocab)-1] + "\n")
    for row in markov.transitions:
      ofs.write(str(row[0]))
      for i in range(1, len(row)):
        ofs.write(", " + str(row[i]))
      ofs.write("\n")
    ofs.close()
	
  # Build Word Markov based on [input file] and [output file]
  if len(args) == 3 and args[0] == "word": #build word Markov
    processed_file = args[1]
    markov_file = args[2]
    markov = buildWordMarkov(markov, processed_file)
    ofs = open(markov_file, "w")
    for i in range(len(markov.vocab)-1):
      ofs.write(markov.vocab[i] + ", ")
    ofs.write(markov.vocab[len(markov.vocab)-1] + "\n")
    for row in markov.transitions:
      ofs.write(str(row[0]))
      for i in range(1, len(row)):
        ofs.write(", " + str(row[i]))
      ofs.write("\n")
    ofs.close()
	
  # Generate char tweet based on default file
  if len(args) == 2 and args[0] == "char": # char [start_char]
    start_char = args[1]
    markov = loadCharMarkov(markov, markov_file)
    tweet = generate_tweet(start_char, 20, markov)
    for char in tweet:
      print(char, sep="", end="")
    print()
	
  # Generate word tweet starting with start_word, default file
  if len(args) == 2 and args[0] == "word": # word [start_word]
    start_word = args[1]
    markov = loadWordMarkov(markov, markov_file)
    tweet = generate_tweet(start_word, 20, markov)
    for char in tweet:
      print(char, " ", end="")
    print()
	
  # Generate tweets for all possible start words based on markov file
  if len(args) == 2 and args[0] == "tweet-word": # tweet-word [markov]
    markov_file = args[1]
    outfile = "tweet-word-smoothed-100-20.txt"
    ofs = open(outfile, "w")
    markov = loadWordMarkov(markov, markov_file)
    for w in markov.vocab:
      if w != "_END_":
        tweet = generate_tweet(w, 20, markov)
        for word in tweet:
          ofs.write(word + " ")
        ofs.write("\n")
    ofs.close()
	
  # Generate tweet based off markov and start_word
  if len(args) == 3 and args[0] == "tweet-word": # tweet-word [markov] [start_word]
    markov_file = args[1]
    start_word = args[2]
    markov = loadWordMarkov(markov, markov_file)
    tweet = generate_tweet(start_word, 20, markov)
    for char in tweet:
      print(char, " ", end="")
    print()
	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  if len(args) == 2 and args[0] == "nltk":
    ifs = open(args[1], "r")
    tokens = nltk.word_tokenize(ifs.readline())
    print("Parts of Speech: ", nltk.pos_tag(tokens))
  

if __name__ == "__main__":
    main()
