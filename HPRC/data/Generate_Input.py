# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import os
import re
import sys
import getopt
import time
import pprint
import nltk


def getSequence(line):
  raw_tokens = nltk.word_tokenize(line)
  raw_tokens = nltk.pos_tag(raw_tokens)
  return raw_tokens

def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
  if len(args) == 1: # in-name
    orig_file = args[0]
    #length = int(args[1])
    #words_file = "words_length" + str(length) + ".txt" #"_" + str(num) + ".txt"
    #tags_file = "tags_length"  + str(length) + ".txt" #"_" + str(num) + ".txt"
    
    max = 71 #max sequence length is 70 (1 char, 1 space)
	
    words_files = [None for x in range(1, max)]
    tags_files = [None for x in range(1, max)]
    ofs_words = [None for x in range(1, max)]
    ofs_tags = [None for x in range(1, max)]
    for i in range(1, max):
      words_files[i-1] = "words_length" + str(i) + ".txt"
      tags_files[i-1] = "tags_length" + str(i) + ".txt"
      ofs_words[i-1] = open(words_files[i-1], "w")
      ofs_tags[i-1] = open(tags_files[i-1], "w")
		
    ifs = open(orig_file, "r")
    lines = ifs.read().split("\n")
    ifs.close()

    for line in lines:
      tokens = getSequence(line)
      if len(tokens) > 0 and len(tokens) < max:
        ofs_w = ofs_words[len(tokens)-1]
        ofs_t = ofs_tags[len(tokens)-1]
        for pair in tokens:
          ofs_words[len(tokens)-1].write(pair[0] + " ")
          ofs_tags[len(tokens)-1].write(pair[1] + " ")
        ofs_words[len(tokens)-1].write("\n")
        ofs_tags[len(tokens)-1].write("\n")
      
    for i in range(1, max):
      ofs_words[i-1].close()
      ofs_tags[i-1].close()


  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  

if __name__ == "__main__":
    main()
