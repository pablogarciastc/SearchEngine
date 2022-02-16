import nltk,math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#!/usr/bin/python

from nltk.stem.porter import PorterStemmer
from math import log10,sqrt
import os
import sys, getopt
import argparse

def get_arg(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str)
    parser.add_argument('-q', type=str)
    parser.add_argument('-qf', type=str)
    parser.add_argument('-rf', type=str)

    args = parser.parse_args()
    if args.c is None:
       exit(2)
    elif args.q is not None:
          return args.c,args.q
    elif args.qf is not None and args.rf is not None:
        return args.c,args.qf,args.rf
    else:
        exit(2)

def main(argv):
    args = get_arg(argv)
    if(len(args)==2):
        corpus = args[0]
        query = args[1]
    elif(len(args)==3):
        corpus = args[0] 
        corpus_path = args[1]
        results_path = args[2]
    else:
        exit(2)


if __name__ == "__main__":
   main(sys.argv[1:])