
import libitg
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
from collections import defaultdict

import numpy as np

LOG_PROB_DEFAULT = [-736.82724089097394, -736.82724089097394]

def get_source_word(fsa: FSA, origin: int, destination: int) -> str:
    """Returns the python string representing a source word from origin to destination (assuming there's a single one)"""
    labels = list(fsa.labels(origin, destination))
    assert len(labels) == 1, 'Use this function only when you know the path is unambiguous, found %d labels %s for (%d, %d)' % (len(labels), labels, origin, destination)
    return labels[0]

def get_target_word(symbol: Symbol):
    """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
    if not symbol.is_terminal():
        raise ValueError('I need a terminal, got %s of type %s' % (symbol, type(symbol)))
    return symbol.root().obj()

def get_bispans(symbol: Span):
    """
    Returns the bispans associated with a symbol. 
    
    The first span returned corresponds to paths in the source FSA (typically a span in the source sentence),
     the second span returned corresponds to either
        a) paths in the target FSA (typically a span in the target sentence)
        or b) paths in the length FSA
    depending on the forest where this symbol comes from.
    """
    if not isinstance(symbol, Span):
        #raise ValueError('I need a span, got %s of type %s' % (symbol, type(symbol)))
        return (None, symbol.obj()), (None, None)
    s, start2, end2 = symbol.obj()  # this unwraps the target or length annotation
    if len(s.obj()) == 3:
        _, start1, end1 = s.obj()  # this unwraps the source annotation
    else: return (None, s.obj()), (None, None)
    return (start1, end1), (start2, end2)

def simple_features(edge: Rule, src_fsa: FSA, eps=Terminal('-EPS-'), 
                    sparse_del=False, sparse_ins=False, sparse_trans=False, sparse_span=False, ibm1_probs=False, embeddings=False, skip_grams=False) -> dict:
    """
    Featurises an edge given
        * rule and spans
        * src sentence as an FSA
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
    crucially, note that the target sentence y is not available!    
    """
    fmap = defaultdict(float)
    if len(edge.rhs) == 2:  # binary rule
        fmap['type:binary'] += 1.0
        # here we could have sparse features of the source string as a function of spans being concatenated
        (ls1, ls2), (lt1, lt2) = get_bispans(edge.rhs[0])  # left of RHS
        (rs1, rs2), (rt1, rt2) = get_bispans(edge.rhs[1])  # right of RHS        
        # TODO: double check these, assign features, add some more
        if ls1 == ls2:  # deletion of source left child
            fmap['order:slcdel'] += 1.0
        if rs1 == rs2:  # deletion of source right child
            fmap['order:srcdel'] += 1.0
        if ls2 == rs1:  # monotone
            fmap['order:mon'] += 1.0
        if ls1 == rs2:  # inverted
            fmap['order:inv'] += 1.0     
    else:  # unary
        symbol = edge.rhs[0]
        if symbol.is_terminal():  # terminal rule
            fmap['type:terminal'] += 1.0
            # we could have IBM1 log probs for the traslation pair or ins/del
            (s1, s2), (t1, t2) = get_bispans(symbol)            
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                # for sure there is a source word
                src_word = get_source_word(src_fsa, s1, s2)                
                fmap['type:deletion'] += 1.0
                # dense versions (for initial development phase)
                if ibm1_probs: 
                    fmap['ibm1:del:logprob'] += ibm1_probs.get(src_word, {}).get('-EPS-', LOG_PROB_DEFAULT)[1]
                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
            else:                  
                # for sure there's a target word
                tgt_word = get_target_word(symbol)
                if s1 == s2:  # has not consumed any source word, must be an eps rule                    
                    fmap['type:insertion'] += 1.0
                    # dense version
                    if ibm1_probs:                        
                        fmap['ibm1:ins:logprob'] += ibm1_probs.get('-EPS-', {}).get(tgt_word, LOG_PROB_DEFAULT)[0]
                    # sparse version
                    if sparse_ins:
                        fmap['ins:%s' % tgt_word] += 1.0
                else:
                    # for sure there's a source word
                    src_word = get_source_word(src_fsa, s1, s2)
                    fmap['type:translation'] += 1.0
                    # dense version
                    if ibm1_probs:
                        fmap['ibm1:x2y:logprob'] += ibm1_probs.get(src_word, {}).get(tgt_word, LOG_PROB_DEFAULT)[0]
                        fmap['ibm1:y2x:logprob'] += ibm1_probs.get(src_word, {}).get(tgt_word, LOG_PROB_DEFAULT)[1]
                    # sparse version                    
                    if sparse_trans:
                        fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0
        else:  # S -> X
            fmap['top'] += 1.0
            
        # span of source and target side 
        (ls1, ls2), (lt1, lt2) = get_bispans(edge.lhs)
        if ls1 == None:
            fmap['root:%s' % (ls2)] += 1
        elif sparse_span:
            fmap['span:source:%i' % (ls2-ls1)] += 1.0
            fmap['span:target:%i' % (lt2-lt1)] += 1.0
        else:
            fmap['span:source'] += ls2-ls1
            fmap['span:target'] += lt2-lt1
            
        # embedding features
        if ls1 != None and embeddings:
            # inside embeddings
            avg_embedding = np.zeros(128)
            skipped = 0
            #print("Span: {}-{}".format(ls1, ls2))
            for offset in range(ls2-ls1):
                #print("Indeces: {}-{}".format(ls1+offset, ls1+offset+1)) 
                src_word = get_source_word(src_fsa, ls1+offset, ls1+offset+1)
                try:
                    avg_embedding += embeddings[src_word]
                except KeyError:
                    print ("Word {} not found in Word2Vec".format(src_word))
                    skipped += 1
            if ls2-ls1-skipped > 0:
                avg_embedding /= (ls2-ls1-skipped)
            for dimension in range(len(avg_embedding)):
                fmap['inside:emb:%i' % (dimension)] += avg_embedding[dimension]
            
            # outside embeddings
            src_len = src_fsa.nb_states()-1
            left = False
            right = False
            if ls1 > 0: left = True
            if ls2 < src_len: right = True
            
            # left
            if left:    
                left_avg_embedding = np.zeros(128)
                skipped = 0
                for offset in range (ls1):
                    src_word = get_source_word(src_fsa, offset, offset+1)
                    try:
                        left_avg_embedding += embeddings[src_word]
                    except KeyError:
                        print ("Word {} not found in Word2Vec".format(src_word))
                        skipped += 1   
                if ls2-ls1-skipped > 0:
                    left_avg_embedding /= (ls1-skipped)
            
            # right
            if right:
                right_avg_embedding = np.zeros(128)
                skipped = 0
                for offset in range (src_len-ls2):
                    src_word = get_source_word(src_fsa, ls2+offset, ls2+offset+1)
                    try:
                        right_avg_embedding += embeddings[src_word]
                    except KeyError:
                        print ("Word {} not found in Word2Vec".format(src_word))
                        skipped += 1   
                if ls2-ls1-skipped > 0:
                    right_avg_embedding /= (src_len-ls2-skipped)
                
            if left and right:
                avg_embedding = (left_avg_embedding + right_avg_embedding) / 2
            elif left:
                avg_embedding = left_avg_embedding
            elif right:
                avg_embedding = right_avg_embedding
                
            for dimension in range(len(avg_embedding)):
                fmap['outside:emb:%i' % (dimension)] += avg_embedding[dimension]
        
        # Skip-Bigram features
        src_len = src_fsa.nb_states()-1
        if ls1 != None and src_len > 1 and skip_grams: 
            first = None
            skip = None
            for offset in range(ls2-ls1):
                src_word = get_source_word(src_fsa, ls1+offset, ls1+offset+1)
                if first != None and skip != None:
                    fmap['skip:%s*%s' % (first, skip)] += 1
                    fmap['skip:%s*%s' % (first, src_word)] += 1                    
                first = skip
                skip = src_word
            fmap['skip:%s*%s' % (first, skip)] += 1                
            
    return fmap