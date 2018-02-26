#!/usr/bin/python
from __future__ import print_function
import sys
import csv
import os
from metrics import precision, recall, f1score, ndcg, average_precision, nnt1, nnt2
import numpy as np
import matplotlib.pyplot as plt


def read_dataset(filename):
    dataset = {}
    with open(filename, 'rb') as fin:
        reader = csv.reader(fin)
        for row in reader:
            fullid = row[0]
            category = row[1]
            subcategory = row[2]
            dataset[fullid] = (category, subcategory)
    return dataset


def load_result(path, filename, queries, targets):
    """Load a retrieval results from file"""
    fullpath = os.path.join(path, filename)
    cutoff = 1000
    r = []
    q = queries[filename[:-4]]
    with open(fullpath, 'rb') as fin:
        for line in fin.readlines()[:cutoff]:
            if line.strip() and not line.startswith('#'): # line is not empty
                retrieved, distance = line.split()
                try:
                    r.append(targets[retrieved])
                except KeyError:
                    continue
    return q, r


def load_results(path, queries, targets):
    """Load all queries from a folder"""
    results = []
    for filename in os.listdir(path):
        try:
            q = queries[filename[:-4]]
        except KeyError:
            continue
        q, r = load_result(path, filename, queries, targets)
        results.append((q, r))
    return results


def freq_count(dataset):
    freqs = {}
    for k, v in dataset.items():
        if v[0] in freqs:
            freqs[v[0]] += 1
        else:
            freqs[v[0]] = 1
    return freqs


def categories_to_rel(queried, retrieved):
    x = []
    for r in retrieved:
        if queried[0] == r[0] and queried[1] == r[1]:
            x.append(2.0)
        elif queried[0] == r[0]:
            x.append(1.0)
        else:
            x.append(0.0)
    return x


def evaluate(path):
    cad = read_dataset('cad.csv')
    rgbd = read_dataset('rgbd.csv')
    freqs = freq_count(cad)
    results = load_results(path, rgbd, cad)
    cutoff = 1000

    mP = 0.0
    mR = 0.0
    mF = 0.0
    mAP = 0.0
    mNDCG = 0.0
    mNNT1 = 0.0
    mNNT2 = 0.0

    for (queried, retrieved) in results:
        x = categories_to_rel(queried, retrieved)[:cutoff]
        f = freqs[queried[0]]
        # Sum up the retrieval scores
        mP += precision(x)
        mR += recall(x, f)
        mF += f1score(x, f)
        mNDCG += ndcg(x)
        mAP += average_precision(x, f)
        mNNT1 += nnt1(x, f)
        mNNT2 += nnt2(x, f)

    n = len(results)
    print('num queries:', n)
    print('mean precision: ', mP / n)
    print('mean recall:', mR / n)
    print('mean F1:', mF / n)
    print('mean AP:', mAP / n)
    print('mean NDCG: ', mNDCG / n)
    print('mean NNT1: ', mNNT1 / n)
    print('mean NNT2: ', mNNT2 / n)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
        stats = evaluate(path)
    else:
        print('Usage: evaluate.py <path_to_results>')
