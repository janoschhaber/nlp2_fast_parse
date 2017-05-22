from formal import *
from alg import * 
from earley import *
from time import time
import libitg
import numpy as np
import sys

from utils import read_lexicon, read_corpus, reduce_corpus, unk
from features import simple_features


def test(lexicon, src_str, tgt_str, inspect_strings=False):

    print('TRAINING INSTANCE: |x|=%d |y|=%d' % (len(src_str.split()), len(tgt_str.split())))

    # Make a source CFG using the whole lexicon
    src_cfg = libitg.make_source_side_finite_itg(lexicon)

    # Make a source FSA
    src_fsa = libitg.make_fsa(src_str)
    #print('SOURCE FSA')
    #print(src_fsa)
    #print()

    # Make a target FSA
    tgt_fsa = libitg.make_fsa(tgt_str)
    #print('TARGET FSA')
    #print(tgt_fsa)
    #print()

    # Intersect source FSA and source CFG
    times = dict()
    times['D(x)'] = time()
    _Dx = libitg.earley(src_cfg, src_fsa, 
            start_symbol=Nonterminal('S'), 
            sprime_symbol=Nonterminal("D(x)"),
            clean=True)  # to illustrate the difference between clean and dirty forests I will disable clean here
    #print(src_forest)
    #print()

    # projection over target vocabulary

    # D(x) is now finite
    Dx_clean = libitg.make_target_side_finite_itg(_Dx, lexicon)
    times['D(x)'] = time() - times['D(x)']
    
    times['D(x,y)'] = time()
    Dxy_clean = libitg.earley(Dx_clean, tgt_fsa,
            start_symbol=Nonterminal("D(x)"), 
            sprime_symbol=Nonterminal('D(x,y)'),
            clean=True)
    times['D(x,y)'] = time() - times['D(x,y)']

    print('D(x) (cleaned): %d rules in %.4f secs' % (len(Dx_clean), times['D(x)']))
    print('D(x,y) (cleaned): %d rules in %.4f secs ' % (len(Dxy_clean), times['D(x,y)']))

    if inspect_strings:
        t0 = time()
        print(' y in clean D(x,y):', tgt_str in libitg.summarise_strings(Dxy_clean, Nonterminal('D(x,y)')))
        print(' gathering strings took %d secs' % (time() - t0))
    
    # and this is how you pickle things
    import dill as pickle
    with open('pickle-test', 'wb') as f:
        pickle.dump(Dxy_clean, f)
    with open('pickle-test', 'rb') as f:
        Dloaded = pickle.load(f)

    print(len(Dloaded), 'loaded')
    print()


if __name__ == '__main__':
    # Test lexicon
    # lexicon = defaultdict(set)
    # lexicon['le'].update(['-EPS-', 'the', 'some', 'a', 'an'])  # we will assume that `le` can be deleted
    # lexicon['e'].update(['-EPS-', 'and', '&', 'also', 'as'])
    # lexicon['chien'].update(['-EPS-', 'dog', 'canine', 'wolf', 'puppy'])
    # lexicon['noir'].update(['-EPS-', 'black', 'noir', 'dark', 'void'])
    # lexicon['blanc'].update(['-EPS-', 'white', 'blank', 'clear', 'flash'])
    # lexicon['petit'].update(['-EPS-', 'small', 'little', 'mini', 'almost'])
    # lexicon['petite'].update(['-EPS-', 'small', 'little', 'mini', 'almost'])
    # lexicon['.'].update(['-EPS-', '.', '!', '?', ','])
    #
    # # lexicon['-EPS-'].update(['.', ',', 'a', 'the', 'some', 'of', 'bit', "'s", "'m", "'ve"])  # we will assume that `the` and `a` can be inserted
    # lexicon['-EPS-'].update(['.', 'a', 'the', 'some', 'of'])  # we will assume that `the` and `a` can be inserted


    lexicon, prob = read_lexicon('data/lexicon', top=5)
    corpus = read_corpus('data/training.zh-en')

    print('LEXICON (excerpt)')
    limit = 10
    counter = 0
    for src_word, tgt_words in lexicon.items():
        print('%s: %s options' % (src_word, tgt_words))
        counter += 1
        if counter == limit: break
    print()
    
    test(lexicon, 
            'le chien noir',
            'black dog',
            inspect_strings=False)
    test(lexicon, 
            'le chien noir',
            'the black dog .',
            inspect_strings=False)
    test(lexicon,
            'le petit chien noir e le petit chien blanc .',
            'the little white dog and the little black dog .')
    test(lexicon,
            'le petit chien noir e le petit chien blanc .',
            'the little white dog and the little black dog .')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien .', 
            'the little black dog and the little white dog and the mini dog .')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit chien petit blanc .', 
            'the little black dog and the little white dog and the mini almost white dog .')
    print('**** The next example should be out of the space of the constrained ITG ***')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien petit blanc e petit noir .', 
            'the little black dog and the little white dog and the dog a bit white and a bit black .')
