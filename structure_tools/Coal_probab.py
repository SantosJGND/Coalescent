import math
import numpy as np


def prob_coal(theta,nsamp):
    
    p= (nsamp - 1) / (nsamp - 1 + theta)
    
    return p

def prob_mut(theta,nsamp):
    
    p= theta / (nsamp - 1 + theta)
    
    return p

###

def Ewens_exact(config_data,theta):
    
    n_samp= sum([(x + 1) * config_data[x] for x in range(len(config_data))])
    
    ThetaN= [theta + x for x in range(len(config_data))]
    ThetaN0= 1
    for y in ThetaN:
        ThetaN0 = ThetaN0 * y
    
    factor_front= math.factorial(len(config_data)) / ThetaN0
    
    classes= 1
    
    for j in range(len(config_data)):
        comb= theta / (j+1)
        comb= comb**config_data[j]
        
        prod= comb * (1 / math.factorial(config_data[j]))
        
        classes= classes * prod
    
    return factor_front * classes

####

def Ewens_recurs(config_vec,theta,prob_array,Pin,prob_bound = 1):
    n_samp= sum([(x + 1) * config_vec[x] for x in range(len(config_vec))])
    
    if config_vec == [1]:
        ## boundary
        
        prob_left= Pin * prob_bound
        
        prob_array.append(prob_left)
        
        return prob_array
    
    if config_vec[0] > 0:
        ## mutation
        prob_left= prob_mut(theta,n_samp)
        
        new_conf= list(config_vec)[:(n_samp - 1)]
        new_conf[0]= new_conf[0] - 1
        
        prob_next= Pin * prob_left
        
        Ewens_recurs(new_conf,theta,prob_array,prob_next)
    
    
    if sum(config_vec[1:]) > 0:
        ## coalesc
        prob_right_I = prob_coal(theta,n_samp)
        
        jsel= [x for x in range(1,len(config_vec)) if config_vec[x] > 0]
        
        for sub in jsel:
            ##  coalesce for all classes still holding more than one allele.
            
            jprop= sub * (config_vec[sub - 1] + 1) / (n_samp - 1)
            
            new_conf= list(config_vec)
            new_conf[sub] -= 1
            new_conf[sub - 1] += 1
            new_conf= new_conf[:(n_samp - 1)]
            
            prob_right= prob_right_I * jprop

            prob_next= Pin * prob_right

            Ewens_recurs(new_conf,theta,prob_array,prob_next)
    
    return prob_array


