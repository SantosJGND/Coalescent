import random
from random import shuffle


import scipy
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import itertools as it
import collections
from IPython.display import clear_output

from Modules_tools import extract_profiles
def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)

def return_fsts(vector_lib,pops):
    
    H= {pop: [1-(vector_lib[pop,x]**2 + (1 - vector_lib[pop,x])**2) for x in range(vector_lib.shape[1])] for pop in pops}
    Store= []
    for comb in it.combinations(pops,2):
        P= [sum([vector_lib[x,i] for x in comb]) / len(comb) for i in range(vector_lib.shape[1])]
        HT= [2 * P[x] * (1 - P[x]) for x in range(len(P))]
        Fst= np.mean([(HT[x] - np.mean([H[p][x] for p in comb])) / HT[x] for x in range(len(P))])
        
        Store.append([comb,Fst])
    
    ### total fst:
    P= [sum([vector_lib[x,i] for x in pops]) / len(pops) for i in range(vector_lib.shape[1])]
    HT= [2 * P[x] * (1 - P[x]) for x in range(len(P))]
    FST= np.mean([(HT[x] - np.mean([H[p][x] for p in pops])) / HT[x] for x in range(len(P))])
    
    return pd.DataFrame(Store,columns= ['pops','fst']),FST



def return_fsts2(freq_array):
    pops= range(freq_array.shape[0])
    H= {pop: [1-(freq_array[pop,x]**2 + (1 - freq_array[pop,x])**2) for x in range(freq_array.shape[1])] for pop in range(freq_array.shape[0])}
    Store= []

    for comb in it.combinations(H.keys(),2):
        P= [sum([freq_array[x,i] for x in comb]) / len(comb) for i in range(freq_array.shape[1])]
        HT= [2 * P[x] * (1 - P[x]) for x in range(len(P))]
        per_locus_fst= [[(HT[x] - np.mean([H[p][x] for p in comb])) / HT[x],0][int(HT[x] == 0)] for x in range(len(P))]
        per_locus_fst= np.nan_to_num(per_locus_fst)
        Fst= np.mean(per_locus_fst)

        Store.append([comb,Fst])
    
    
    ### total fst:
    P= [sum([freq_array[x,i] for x in pops]) / len(pops) for i in range(freq_array.shape[1])]
    HT= [2 * P[x] * (1 - P[x]) for x in range(len(P))]
    FST= np.mean([(HT[x] - np.mean([H[p][x] for p in pops])) / HT[x] for x in range(len(P))])
    
    return pd.DataFrame(Store,columns= ['pops','fst'])


def number_coord(Numbers_box,Nrow= 28,Ncol= 28,Height= 30,Length= 30):

    kde_store= {}

    Hstep= Height / float(Nrow)
    Lstep= Length / float(Ncol)

    range_height= [0,Height]
    range_length= [0,Length]

    ## Now, MNIST images are 28 x 28. lets assume that's height x width in increasing order.
    ## then, each 28 elements represent a row, starting from the bottom.
    ## each row will comprise 1 / 28 th of our height.
    ## each column will comprise 1 / 28 th of our length.

    for chiffre in Numbers_box.keys():

        Bit_image= Numbers_box[chiffre]['image']
        labs= Numbers_box[chiffre]['label']

        Bit_image= np.array(Bit_image).reshape(Nrow,Ncol)

        ## coordintes of positive values
        coordinates_positive= np.array(np.where(Bit_image > 0)).T

        ## get their density on our layout
        datum= []
        dotum= []

        for l in range(coordinates_positive.shape[0]):
            coords= coordinates_positive[l,:]
            N= Bit_image[coords[0],coords[1]]

            for s in range(1):
                datum.append([coords[0] * Hstep, coords[1] * Lstep])
                dotum.append([coords[0] * Hstep, coords[1] * Lstep])

        datum= np.array(datum)
        dotum= np.array(dotum)

        kde_store[labs]= datum
    
    print(datum.shape)
    return kde_store



def plot_number(kde_store= {},plot_who= [],Height= 30,Length= 30,P= 70,trigger_warning= .6,param_grid_I= 0.2,param_grid_II=.4,steps= 20):
    kde_dict= {}

    range_height= [0,Height]
    range_length= [0,Length]
    
    
    for numb in kde_store.keys():


        datum= kde_store[numb]
        params_dens= {'bandwidth': np.linspace(param_grid_I, param_grid_II,steps)}
        grid_dens = GridSearchCV(KernelDensity(algorithm = "ball_tree",breadth_first = False), params_dens,verbose=0)

        traces= [x for x in it.product(range(P),range(P))]

        i_coords, j_coords = np.meshgrid(np.linspace(range_height[0],range_height[1],P),
                              np.linspace(range_length[0],range_length[1],P),indexing= 'ij')

        background= np.array([i_coords, j_coords])

        background= [background[:,c[0],c[1]] for c in traces]
        background=np.array(background)


        ### Density measure
        grid_dens.fit(datum)
        kde = grid_dens.best_estimator_

        P_dist= kde.score_samples(datum)
        scores= kde.score_samples(background)

        scores= np.exp(scores)
        scores= np.array([x for x in scipy.stats.norm(np.mean(scores),np.std(scores)).cdf(scores)])

        ### haplotypes measure
        datum= np.unique(datum,axis= 0)

        grid_dens.fit(datum)
        kde = grid_dens.best_estimator_

        P_dist= kde.score_samples(datum)
        scores_haps= kde.score_samples(background)
        scores_haps= np.exp(scores_haps)

        kde_dict[numb]= {
            'kde': kde,
            'scores': scores_haps
        }

        if numb in plot_who:

            #scores_combine= scipy.stats.norm(np.mean(scores_haps),np.std(scores_haps)).cdf(scores_haps)

            scores_combine= scores_haps / max(scores_haps)


            if trigger_warning:
                scores_combine[scores_combine > trigger_warning] = 1

            fig= [go.Scatter3d(
                x= background[:,0],
                y= background[:,1],
            #    z= scores[[x for x in range(len(scores)) if scores[x] > 0]],
                z= scores_combine,
                mode= 'markers',
                marker= {
                    'color':scores_combine,
                    'colorbar': go.ColorBar(
                        title= 'ColorBar'
                    ),
                    'colorscale':'Viridis',
                    'line': {'width': 0},
                    'size': 4,
                    'symbol': 'circle',
                  "opacity": 1
                  }
            )]


            fig = go.Figure(data=fig)
            iplot(fig)
    
    return kde_dict



def get_freqs(Pops,features,coords,range_dist= [0,10],
                            step_dist= .1,
                            total_range= 1000,
                            diff_pattern= '',
                            target= [0,1]):

    fst_labels= []
    Fsts_crawl= []
    angle_list= []
    Distances_crawl= []

    for angle in np.arange(range_dist[0],range_dist[1],step_dist):
        coords= features[Pops,:]
        vector2= coords[target[0]] - coords[target[1]]

        if diff_pattern == 'sinusoidal':
            coords[target[0]] = coords[target[0]] + [sin(angle) * x for x in vector2]
        if diff_pattern == 'linear':
            coords[target[0]] = coords[target[0]] - [(angle- range_dist[0]) / total_range * x for x in vector2]
        else:
            coords= coords


        new_freqs= pca.inverse_transform(coords)

        scramble= [x for x in range(new_freqs.shape[1])]
        shuffle(scramble)

        new_freqs= new_freqs[:,scramble]


        new_freqs[new_freqs > 1] = 1
        new_freqs[new_freqs < 0] = 0

        Pairwise= return_fsts2(new_freqs)

        Distances= []
        for train in it.combinations([x for x in range(new_freqs.shape[0])],2):
            Distances.append(np.sqrt((coords[train[0]][0] - coords[train[1]][0])**2 + (coords[train[0]][1] - coords[train[1]][1])**2) + (coords[train[0]][2] - coords[train[1]][2])**2)
        Distances_crawl.extend(Distances)

        fst_labels.extend(Pairwise.pops)

        Fsts_crawl.extend(Pairwise.fst)
        angle_list.extend([angle] * Pairwise.shape[0])


    Control= np.array([angle_list,Fsts_crawl]).T
    
    return Control, Fsts_crawl, angle_list




def generate_samples_digits(features,Whose,coords,Origins,ind_to_group,Pop_to_kde,kde_store,Pops,pca_obj,Chr= 1,
                                L= 5000,
                                Height= 30,
                                Length= 30,
                                range_dist= [0,10],
                                total_range= [],
                                step_dist= .05,
                                window_length= 5000,
                                trigger_warning= 0.6,
                                diff_pattern= '',
                                select_pop= [0,1],
                                labels= [0,1,2],
                                label_vector= [],
                                target= [0,1],
                                color_ref= [],
                                N_pops= 2,
                                COp= 1):

    label_indicies= {x:[y for y in range(len(label_vector)) if label_vector[y] == x] for x in Origins.keys()}

    Windows= recursively_default_dict()
    Blocks_truth= recursively_default_dict()
    Haplotypes= recursively_default_dict()

    Ideo= []

    Fst_windows= []
    Fst_crawl= []
    Fst_labels= []

    target_indx= {z:[x for x in range(len(label_vector)) if label_vector[x] == z] for z in target}

    current= recursively_default_dict()
    d= 0

    for angle in np.arange(range_dist[0],range_dist[1],step_dist):
        print(angle)
        coords= features[Pops,:]
        vector2= coords[target[0]] - coords[target[1]]

        if diff_pattern == 'sinusoidal':
            coords[target[0]] = coords[target[0]] + [sin(angle) * x for x in vector2]
        if diff_pattern == 'linear':
            coords[target[0]] = coords[target[0]] - [(angle - range_dist[0]) / total_range * x for x in vector2]
        else:
            coords= coords
    
        new_freqs= pca_obj.inverse_transform(coords)
        bl= int(angle*10000)
        end= bl+ 999


        scramble= [x for x in range(new_freqs.shape[1])]
        shuffle(scramble)

        new_freqs= new_freqs[:,scramble]

        ##### modify the transition probabilities of pop1 samples as in the above plot:
        for popeye in select_pop:
            order= list(Origins[popeye].keys())

            for indy in range(len(order)):
                pos= order[indy]
                layout_coords= [pos * (Height / float(len(Origins[popeye]))),((angle - range_dist[0]) / (total_range)) * Length]
                layout_coords= np.array(layout_coords).reshape(1,-1)

                pop_kde= Pop_to_kde[popeye]

                Prob_1= kde_store[pop_kde]['kde'].score_samples(layout_coords)
                Prob_1= np.exp(Prob_1)[0]
                Prob_1= Prob_1 / max(kde_store[pop_kde]['scores'])

                if trigger_warning and Prob_1 >= trigger_warning:

                    Prob_1= 1


                Origins[popeye][pos][popeye]= 1 - Prob_1
                Origins[popeye][pos][2]= Prob_1


        data= []
        local_labels= []

        for acc in range(len(Whose)):
            Subject = 'sample' + str(acc)

            transition_p= Origins[ind_to_group[acc][0]][ind_to_group[acc][1]]

            if current[acc]:
                cross_over= np.random.choice([0,1], p=[1-COp,COp])
                if cross_over == 1:
                    k= np.random.choice(labels, p=transition_p)
                    current[acc]= k
                else:
                    k= current[acc]
            else:
                k= np.random.choice(labels, p=transition_p)
                current[acc]= k

            probs= new_freqs[k,:]

            probs[(probs > 1)]= 1
            probs[(probs < 0)]= 0

            Haps= [np.random.choice([1,0],p= [1-probs[x],probs[x]]) for x in range(L)]

            Stock = ['Region_'+str(Chr)+ '_' + Subject,int(d*window_length),end,color_ref[k]]
            Ideo.append(Stock)
            data.append(Haps)
            local_labels.append(k + 1)

        data= np.array(data)

        Haplotypes[Chr][int(d*window_length)]= data

        pca2 = PCA(n_components=3, whiten=False,svd_solver='randomized')

        data= pca2.fit_transform(data)

        profiles= extract_profiles(data,target_indx)

        ### get population fsts
        Pairwise= return_fsts2(new_freqs)
        #Fst_labels.extend(Pairwise.pops)

        #Fst_crawl.extend(Pairwise.fst)

        #Fst_windows.extend([bl] * Pairwise.shape[0])
        ### store stuff.
        Blocks_truth[Chr][d*window_length]= local_labels
        Windows[Chr][d*window_length]= profiles

        d += 1
        clear_output()



    Out= {
        x: {bl: bl+window_length-1 for bl in Windows[x].keys()} for x in Windows.keys() 
         }
    
    return Blocks_truth, Windows, Out, Haplotypes, Ideo
