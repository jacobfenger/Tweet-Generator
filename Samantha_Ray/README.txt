Samantha Ray (823004053) Markov Code

Sample data files have been included in the "data" folder. 
Each program assumes that the data files are in the same directory;
the files were moved into a sub-folder for the sake of decluttering.

Naming convention:
Each words and tags file of a given length are a pair
words_length##.txt -> sequence of words on each line
tags_length##.txt  -> corresponding tag sequence on each line

All programs use Python3 and numpy.

------------------------------------------------

Markov Chain

Description: Generates 1000 sequences of length [sequence length] and writes to file
MarkovOut_lengthXX_sizeXX.txt

Run command: python Markov_mp.py [sequence length] [number of tweets to train on]
>>> e.g. python Markov_mp.py 20 10000 

------------------------------------------------

Hidden Markov Model

Description: Generates a sequence of characters starting with [start_char] of length [length]
with transition states of length [k]. Random is a 0/1 flag to determine to seed the last character
in the sequence as the end_symbol or a random character

Run command: python Viterbi_char.py [train sequence length] [number of tweets] [k] [start_char] [length] [random (optional)]
>>> e.g. python Viterbi_char.py 5 5000 2 i 5 random
         python Viterbi_char.py 5 5000 2 i 5

------------------------------------------------

Reverse Viterbi - Bigram

Description: Generates a sequence of words based on given tag sequence
Uses multiprocessing and writes 10 output sequences to file
BigramOut_length##_size##.txt

Run command: python Viterbi_bigram_mp.py [sequence length] [number of tweets]
>>> e.g. python Viterbi_bigram_mp.py 5 10000

Batch file included: Bigram5.slurm

------------------------------------------------

Reverse Viterbi - Generalized

Description: Builds a trellis of given [sequence length] from [number of tweets] input samples
or loads trellis and generates sequences based on given final tag 
Uses multiprocessing and writes all possible output sequences to file (or [NONE] if the sequence had 0 probability)
generalOut_mp_length##_size##.txt
trellis_mp_length##_size##.txt
bp_mp_length##_size##.txt

Run command: python Viterbi_generalized_mp.py ["build"/"load"] [sequence length] [number of tweets]
>>> e.g. python Viterbi_generalized_mp.py build 5 10000
         python Viterbi_generalized_mp.py load 5 10000

Batch file included: General_mp5.slurm

------------------------------------------------

Reverse Viterbi - Filtered

Description: Builds a trellis of given [sequence length] from [number of tweets] input samples
with vocabulary filtered to only include words with [threshold] counts
Uses multiprocessing and writes 10 output sequences to file
filteredOut_mp_length##_size##.txt

Run command: python Viterbi_filtered_mp.py [sequence length] [number of tweets] [threshold]
>>> e.g. python Viterbi_filtered_mp.py 5 10000 2

Batch files included: Filtered2_5.slurm, Filtered3_5.slurm, Filtered10_5.slurm

------------------------------------------------