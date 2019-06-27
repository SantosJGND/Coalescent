import math
import numpy as np
import random

import collections

def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)


import time
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import MeanShift, estimate_bandwidth

def gem_sampler(root_lib,point_up,range_theta,theta_array= [],max_time= 4e5,
            Ngaps= 3,sink= 0,permN= 1000,Ave_vec= [2.12],sig= 1,step= .1,rand_comb= False):
    
    from structure_tools.Coal_index import theta_time, theta_function, tree_ascent_times
    
    comb_theta= []
    comb_likes= []
    #
        
    if len(Ave_vec)==1:
        Ave_vec= Ave_vec * Ngaps

    combi= Ave_vec
    
    if rand_comb:
        combi= list(np.random.choice(list(range_theta),Ngaps))
    
    if len(theta_array) != Ngaps:
        theta_array= theta_time(list(combi),max_time,Ngaps)
    
    node_weigths, paths_backward, node_times = tree_ascent_times(root_lib,point_up,sink,
                                                                 mu= 9e-8,theta_time_array= theta_array)
    
    prob= np.sum(node_weigths[sink][0])
    prob= round(prob,5)
    comb_theta.append(combi)
    comb_likes.append(prob)
    last_comb= list(combi)
    last_prob= prob
    comb_diff_I= random.randint(0,len(combi)-1)
    #
    Ave= Ave_vec[comb_diff_I]
    combi[comb_diff_I]+= [-1,1][int(combi[comb_diff_I]) < Ave] * step
    
    for i in range(permN):
        
        if len(comb_likes) > 10:
            if len(list(set(comb_likes[(len(comb_likes) - 10):]))) == 1:
                combi= list(last_comb)
                
                theta_array= theta_time(list(combi),max_time,Ngaps)

                node_weigths, paths_backward, node_times = tree_ascent_times(root_lib,point_up,sink,
                                                                             mu= 9e-8,theta_time_array= theta_array)
                prob= np.sum(node_weigths[sink][0])
                prob= round(prob,5)
                
                Theta_lib= {
                        'probs': prob,
                        'times': node_times,
                        'comb': theta_array
                    }

                return Theta_lib, comb_likes, comb_theta
        
        
        theta_array= theta_time(list(combi),max_time,Ngaps)
        
        node_weigths, paths_backward, node_times = tree_ascent_times(root_lib,point_up,sink,
                                                                     mu= 9e-8,theta_time_array= theta_array)
        
        prob= np.sum(node_weigths[sink][0])
        prob= round(prob,5)
        #
        last_prob= comb_likes[-1]
        #
        if prob > last_prob:
            
            comb_theta.append(combi)
            comb_likes.append(prob)
            last_prob= prob
            comb_diff= [combi[x] - last_comb[x] for x in range(len(combi))]
            last_comb= list(combi)
            combi= [combi[x] + comb_diff[x] for x in range(len(combi))]
        
        elif prob == last_prob:
            
            comb_theta.append(combi)
            comb_likes.append(prob)
            last_comb= list(combi)
            last_prob= prob
            comb_diff_I= random.randint(0,len(combi) - 1)
            #
            Ave= Ave_vec[comb_diff_I]
            combi[comb_diff_I]+= [-1,1][int(combi[comb_diff_I]) < Ave] * step
            #
        
        elif prob < last_prob:
            comb_diff_I= random.randint(0,len(combi)-1)
            combi= list(last_comb)
            
            Ave= Ave_vec[comb_diff_I]
            combi[comb_diff_I]+= [-1,1][int(combi[comb_diff_I]) < Ave] * step
    
    
    theta_array= theta_time(list(combi),max_time,Ngaps)

    node_weigths, paths_backward, node_times = tree_ascent_times(root_lib,point_up,sink,
                                                                 mu= 9e-8,theta_time_array= theta_array)
    
    prob= np.sum(node_weigths[sink][0])
    prob= round(prob,5)
    
    Theta_lib= {
            'probs': prob,
            'times': node_times,
            'comb': theta_array
        }
        
    return Theta_lib, comb_likes, comb_theta


from IPython.display import clear_output


###########
########### PCA optimization


def pca_optimize(feats_combi,data_combs,Z_vec,pca_ob,
			root_lib,point_up,sink,
			N_samps= 50,Nlayers= 10,max_time= 4e5, 
			Ngaps= 5,Ncomps= 4,thresh_z= 2,wait_p= 10):
    
    from structure_tools.Coal_index import theta_time, theta_function, tree_ascent_times
    from sklearn.decomposition import PCA
    from IPython.display import clear_output

    prob_mean= []
    prob_median= []
    prob_sd= []
    
    waited= 0
    for layer in range(Nlayers):
        
        clear_output()
        
        print(feats_combi.shape)
        Z_ch= list(Z_vec.reshape(1,-1)[0])
        Z_ch= np.argsort(Z_ch)
        
        if len(Z_ch) < 15:
            return prob_mean, prob_median, prob_sd, new_data, Theta_record
        
        Z_ch= Z_ch[(len(Z_ch) - 15):]
        #Z_ch= [x for x in range(len(Z_vec)) if Z_vec[x] >= 1]
        
        Z_high= feats_combi[Z_ch]

        print(Z_high.shape)
        params = {'bandwidth': np.linspace(0.1, 2, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
        grid.fit(Z_high)
        
        kde = grid.best_estimator_
        
        new_data = kde.sample(N_samps, random_state=0)
        
        Z_test= kde.score_samples(Z_high)
        #print(Z_test)
        Z_test= np.exp(Z_test)

        #Z_test= np.where(Z_test > 0.005)[0]
        new_data = pca_ob.inverse_transform(new_data)
        new_data[new_data < 0]= 0.1
        
        Theta_record= recursively_default_dict()
        
        #print(new_data[0])
        if not len(prob_mean):
            min_p= 0.00001
        
        print(min_p)
        
        for combo in range(new_data.shape[0]):

            combi= new_data[combo]
            
            theta_array= theta_time(list(combi),max_time,Ngaps)

            node_weigths, paths_backward, node_times = tree_ascent_times(root_lib,point_up,sink,
                                                                         mu= 9e-8,theta_time_array= theta_array)
            
            
            if node_weigths[sink][0] >= min_p and node_weigths[sink][0] < 1:
                Theta_record[tuple(list(combi))]= {
                    'probs': node_weigths[sink][0],
                    'times': node_times,
                    'comb': theta_array
                }
        
        probs_keys= list(Theta_record.keys())
        probs_vector= [Theta_record[th]['probs'] for th in probs_keys]
        probs_vector= np.array(probs_vector).reshape(-1,1)
        
        mean_p= np.mean(probs_vector)
        median_p= np.median(probs_vector)
        print(median_p)
        
        if len(prob_median):            
            if median_p <= prob_median[-1]:
                if waited >= wait_p:
                    return prob_mean, prob_median, prob_sd, new_data, Theta_record
                waited += 1
                continue
        
        waited= 0
        
        prob_mean.append(mean_p)
        
        prob_median.append(median_p)
        prob_sd.append(np.std(probs_vector))
        
        min_p= prob_mean[-1] - (prob_sd[-1] / 2)
        
        Z_vec= (probs_vector - np.mean(probs_vector)) / np.std(probs_vector)
        
        data_combs= data_combs[Z_ch]
        new_th= [x for x in probs_keys]
        new_th= np.array(new_th)
        
        data_combs= np.vstack((data_combs,new_th))
        
        print(data_combs.shape)

        ## Perform PCA
        Ncomps= [data_combs.shape[1],5][int(data_combs.shape[1] > 5)]
        pca_ob = PCA(n_components=Ncomps, whiten=False,svd_solver='randomized').fit(data_combs)
        
        feats_combi= pca_ob.transform(data_combs)
    
    return prob_mean, prob_median, prob_sd, new_data, Theta_record


