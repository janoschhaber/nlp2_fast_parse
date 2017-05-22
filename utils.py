import numpy as np
from collections import defaultdict


def read_lexicon(path, top=10, top_eps=50):
    with open(path, 'r') as f:
        l = f.readlines()

    m = defaultdict(list)
    for e in l:
        e = e.split()
        e[0] = e[0] if e[0] != '<NULL>' else '-EPS-'
        e[1] = e[1] if e[1] != '<NULL>' else '-EPS-'
        e[2] = e[2] if e[2] != 'NA' and float(e[2]) > 1e-320 else 1e-320
        e[3] = e[3] if e[3] != 'NA' and float(e[3]) > 1e-320 else 1e-320
        m[e[0]].append((e[1], np.log(float(e[2])), np.log(float(e[3]))))
    
    m = {k: sorted(v, key=lambda x: x[1] + x[2], reverse=True) for k, v in m.items()}
    q = {k: {e[0]: (e[1], e[2]) for e in v} for k, v in m.items()}

    m = {k: v[:min(len(v), top - 1)] + ['-EPS-'] if k != '-EPS-' else v[:min(len(v), top_eps)] for k, v in m.items()}
    m = {k: set([e[0] for e in v]) for k, v in m.items()}
    
    m['-UNK-'] = set({'-UNK-'})
    m['-EPS-'].add('-UNK-')
    
    if '-EPS-' in m['-EPS-']:
        m['-EPS-'].remove('-EPS-')
    
    return m, q

def read_corpus(path):
    with open(path, 'r') as f:
        l = f.readlines()
        
    return [e.strip().split(' ||| ') for e in l]

def reduce_corpus(corpus, start=0, N=10, L=10):
    new_corpus = []
    i = 0
    while len(new_corpus) < N:
        if len(corpus[start + i][1].split()) < L:
            new_corpus.append(corpus[start + i])
        i += 1

    return new_corpus

def unk(s, l):
    return ' '.join([e if e in l else '-UNK-' for e in s.split()])
