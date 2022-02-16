#!/usr/bin/python

from math import log10, sqrt
import os
import sys
import getopt
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
        return args.c, args.q
    elif args.qf is not None and args.rf is not None:
        return args.c, args.qf, args.rf
    else:
        exit(2)


def cf(queries, results):
    # Se comprueba si el segundo es None para ver
    # si es un modo u otro, quizás hay alguna forma mejor de hacerlo
    if results is None:
        print("Modo Query unica")
    else:
        print("Modo Batch")
    return


def moocs(queries, results):
    if results is None:
        print("Modo Query unica")
    else:
        print("Modo Batch")
    return


def main(argv):
    args = get_arg(argv)
    if(len(args) == 2):
        corpus = args[0]
        query = args[1]
        if corpus == "cf":
            cf(query, None)
        if corpus == "moocs":
            moocs(query, None)
    elif(len(args) == 3):
        corpus = args[0]
        queries_path = args[1]
        results_path = args[2]
        if corpus == "cf":
            cf(queries_path, results_path)
        if corpus == "moocs":
            moocs(queries_path, results_path)
    else:
        exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
