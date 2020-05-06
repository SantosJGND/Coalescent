from scipy.stats import binom
import numpy as np
import itertools as it
import math

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def get_grid(P= 20,range_height= [0,1],range_length= [0,1]):

    traces= [x for x in it.product(range(P),range(P))]

    i_coords, j_coords = np.meshgrid(np.linspace(range_height[0],range_height[1],P),
                          np.linspace(range_length[0],range_length[1],P),indexing= 'ij')

    traces= [x for x in it.product(range(P),range(P))]

    i_coords, j_coords = np.meshgrid(np.linspace(range_height[0],range_height[1],P),
                          np.linspace(range_length[0],range_length[1],P),indexing= 'ij')

    background= np.array([i_coords, j_coords])

    background= [background[:,c[0],c[1]] for c in traces]
    background=np.array(background)
    
    return background,i_coords, j_coords


def single_gen_matrix(Ne= 1092, ploidy= 2,precision= 1092,s=0):
    '''
    define transition probabilities for alleles at any given frequency from 1 to Ne.
    '''
    pop_size= Ne * ploidy
    space= np.linspace(0,pop_size,precision,dtype=int)
    t= np.tile(np.array(space),(precision,1))
    
    probs= [x / (pop_size) for x in space]
    
    ## 
    sadjust= [x * (1+s) for x in probs]
    scomp= [(1-x) for x in probs]
    
    new_probs= [sadjust[x] / (scomp[x]+sadjust[x]) for x in range(precision)]
    new_probs= np.nan_to_num(new_probs)
    
    probs= new_probs
    
    freq_matrix= binom.pmf(t.T,pop_size,probs)
    
    return freq_matrix


def freq_progression(fr= 1,n_gens= 20,freq_matrix= {},remove_tails= False):
    '''frequency distribution after n generations given initial freq'''
    fixed_tally= []
    if isinstance(fr,int):
        freq_ar= [0] * freq_matrix.shape[0]
        freq_ar[fr]= 1
    else:
        freq_ar= fr
    
    for idx in range(n_gens):
        
        freq_ar= np.array(freq_ar) @ freq_matrix.T
        freq_ar= freq_ar.reshape(1,-1)
        
        prop_fixed= sum([freq_ar[0,0],freq_ar[0,-1]])
        prop_fixed= prop_fixed / np.sum(freq_ar,axis= 1)
        fixed_tally.append(prop_fixed[0])
        
        if remove_tails:
            freq_ar[0,0]= 0
            freq_ar[0,-1]= 0
        
        #print(freq_ar.shape)
            
        freq_ar= freq_ar.T / np.sum(freq_ar,axis= 1)
        
        freq_ar= freq_ar.T
    
    return freq_ar, fixed_tally
