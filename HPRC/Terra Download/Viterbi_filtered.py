# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import os
import re
import sys
import getopt
import time
import pprint
import math
	
class Markov:
  def __init__(self):
    self.vocab_states = {} # map vocab to index in transitions and  emissions
    self.tags_states = {}  # map tag to index in emissions
    self.vocab_counts = {} # map words to number of occurrences
    
    self.transitions = []  # 2D array - prob that word follows a word
    self.emissions = []    # 2D array - prob of tag given word
    self.vocab = []        # Words based on NLTK parsing
    self.tags = []         # NLTK tags
    self.tag_vocab = []    # 2D array - each tag has an individual vocab associated with it
    
    self.vocab_size = 0
    self.tags_size = 0
    
    self.start_symbol = "_START_"
    self.end_symbol = "_END_"
    self.unknown_symbol = "_???_"
	
  def initialize(self, tweet_lines, tag_lines):
    #self.vocab = [self.start_symbol, self.end_symbol, self.unknown_symbol] #assumption that no tweet has these words
    for line in tweet_lines:
      line = line.strip().split(" ")
      for ele in line:
        if ele not in self.vocab:
          self.vocab += [ele]
          self.vocab_states[ele] = self.vocab_size
          self.vocab_counts[ele] = 0
          self.vocab_size += 1
        self.vocab_counts[ele] += 1
    for line in tag_lines:
      line = line.strip().split(" ")
      for ele in line:
        if ele not in self.tags:
          self.tags += [ele]
          self.tags_states[ele] = self.tags_size
          self.tags_size += 1
    		  	  
    print("Before filtering: ", self.vocab_size)
    # NEW - filter out rare words to reduce noise (goal: vocab around 700-1000 words)
    num_removed_words = self.filterWordsCount(2) # only keep words with at least n occurences
    print("After filtering to 2 counts: ", self.vocab_size)
	
    # Add special symbols to dictionaries
    self.vocab.append(self.start_symbol)
    self.vocab.append(self.end_symbol)
    self.vocab.append(self.unknown_symbol)
    self.vocab_states[self.start_symbol] = self.vocab_size
    self.vocab_states[self.end_symbol] = self.vocab_size + 1
    self.vocab_states[self.unknown_symbol] = self.vocab_size + 2
    self.vocab_counts[self.start_symbol] = len(tweet_lines)
    self.vocab_counts[self.end_symbol] = len(tweet_lines)
    self.vocab_counts[self.unknown_symbol] = num_removed_words
    self.vocab_size += 3
  
    # Transitions: Word_i | Word_(i-1)
    self.transitions = np.zeros((self.vocab_size, self.vocab_size))
    # Emissions: Tag | Word
    self.emissions = np.zeros((self.tags_size, self.vocab_size))
    # Tag Vocab:
    self.tag_vocab = [[] for x in range(self.tags_size)]
    pass
	
  def buildMarkov(self, tweet_file, tag_file):
    #Initialize Markov
    ifs = open(tweet_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tweet_lines = ifs.read().split("\n")
    ifs.close()
    ifs = open(tag_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tag_lines = ifs.read().split("\n")
    ifs.close()
    self.initialize(tweet_lines, tag_lines)
  
    # Fill in transitions and emissions
    for index in range(len(tweet_lines)):
      if tweet_lines[index] != "":
        tokens = tag_lines[index].strip().split(" ")
        words = self.convertUnknownWords(tweet_lines[index].strip().split(" ")) #NEW
		
        prev = self.vocab_states[self.start_symbol] #index of "start" symbol
        for i in range(len(words)): #looping through tags in tokens as well
          curr = self.vocab_states[words[i]]  # index of current word
          tag = self.tags_states[tokens[i]]   # index of current tag
          self.transitions[prev][curr] += 1
          self.emissions[tag][curr] += 1
          if words[i] != self.unknown_symbol:
            self.tag_vocab[tag].append(words[i]) # add word to tag vocab, keeping duplicates
          prev = curr
        curr = self.vocab_states[self.end_symbol] #index of "end" symbol
        self.transitions[prev][curr] += 1
    
    for i in range(self.vocab_size):
      self.transitions[i] /= self.vocab_counts[self.recoverWord(i)]
	
    for j in range(self.tags_size):
      for i in range(self.vocab_size):
        self.emissions[j][i] /= self.vocab_counts[self.recoverWord(i)]
    pass
	
  def convertUnknownWords(self, line):
    for i in range(len(line)):
      if line[i] not in self.vocab:
        line[i] = self.unknown_symbol #unknown word symbol
    return line
	
  def filterWordsCount(self, threshold):
    new_vocab = []
    num = 0
    new_vocab_counts = {}
    for word in self.vocab:
      if self.vocab_counts[word] >= threshold:
        new_vocab.append(word)
        new_vocab_counts[word] = self.vocab_counts[word]
      else:
        num += self.vocab_counts[word]
    self.updateVocab(new_vocab, new_vocab_counts)
    return num
		  
	  
  def updateVocab(self, new_vocab, new_vocab_counts):
    self.vocab_states = {} # reset dictionary
    self.vocab_size = 0
    self.vocab = new_vocab
    self.vocab_counts = new_vocab_counts
    for w in self.vocab:
      self.vocab_states[w] = self.vocab_size
      self.vocab_size += 1
    pass
        
	
  def recoverWord(self, index):
    for w in self.vocab:
      if(self.vocab_states[w] == index):
        return w
    return None
	
  # Reversed Viterbi
  def generate_tweet(self, sequence):
    n = len(sequence)
    trellis = np.zeros(( n + 1, self.vocab_size))
    bp = np.zeros(( n + 1, self.vocab_size))
    start = self.vocab_states[self.start_symbol] #index of "start" symbol
  
    #Initialize
    trellis[0][start] = 1
	
    #"Recursion"
    for i in range(1, n + 1):
      tag = self.tags_states[sequence[i-1]] # -1 because including _START_ at front
      for word in range(self.vocab_size):
        for j in range(self.vocab_size):
          temp = trellis[i-1][j] * self.transitions[j][word]
          if(temp > trellis[i][word]):
            trellis[i][word] = temp
            bp[i][word] = j
        trellis[i][word] *= self.emissions[tag][word]

    w_max = 0
    vit_max = 0
    end = self.vocab_states[self.end_symbol]
    # Find best final word in order to go backwards
    for word in range(self.vocab_size):
      if(trellis[n][word]*self.transitions[word][end] > vit_max):
        w_max = word
        vit_max = trellis[n][word]*self.transitions[word][end]
	  
    result = [None for k in range(n)]
    i = n
    w = w_max
    while i > 0:
      word = self.recoverWord(w)
      if word == self.unknown_symbol: # unknown word symbol
        # Choose a random word that has been associated with the current tag in the sequence
        word = rm.choice(self.tag_vocab[self.tags_states[sequence[i-1]]])
      result[i-1] = word
      w = int(bp[i][w])
      i -= 1
    return result



def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
    
  if len(args) == 1: # length
    #tweet_file = args[0]
    #sequence_file = tweet_file.split(".")[0] + "_sequences.txt"
    #output_file = tweet_file.split(".")[0] + "_filteredOutput.txt"
    
    length = int(args[0])
    words_file = "words_length" + str(length) + ".txt"
    tags_file = "tags_length" + str(length) + ".txt"
    output_file = "filteredOut_length" + str(length) + ".txt"
    
    markov = Markov()
    markov.buildMarkov(words_file, tags_file)
    print("\nBuild Time = ", (time.time() - time_start), " s")
    print("vocab size = ", markov.vocab_size)
	
    ifs = open(tags_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    lines = ifs.read().split("\n")
    ifs.close()
    tag_sequences = [None for x in range(len(lines))]
    for i in range(len(lines)):
      tag_sequences[i] = lines[i].strip().split(" ")
	
	
    ofs = open(output_file, "w")
    #debug = 0
    for tag_sequence in tag_sequences:
      tweet = markov.generate_tweet(tag_sequence)
      for i in range(len(tweet)):
        ofs.write(tweet[i] + " ")
      ofs.write("\n")
      #debug += 1
      #if debug == 10:
      #  break
    ofs.close()
  	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
