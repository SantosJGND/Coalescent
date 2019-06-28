
import time
from sklearn.metrics import pairwise_distances
 

import collections

def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)


import numpy as np 
import itertools as it 


def Inf_sites_graph(Dict_mat,point_up,layer_range= 10, print_layer= True,print_time= True):
    
    t1 = time.time()
   
    MRCA= False
    
    layer= 0
    
    leaves= {}
    graph_dict= {}
    
    while MRCA == False:
        if MRCA:
            t2 = time.time()
            tscale= 's'
            tpass= t2 - t1

            if tpass > 600:
                tpass = tpass / 60
                tscale= 'm'

            tpass= round(tpass,3)
            if print_time:
                
                print('time elapsed: {} {}'.format(tpass,tscale))

            return Dict_mat, point_up
        
        
        if print_layer:
            print('layer: {}; len: {}'.format(layer,len(Dict_mat[layer])-1))
        
        if len(Dict_mat[layer]) == 1:
            #if sum(Dict_mat[layer][0][:,1]) == 1:
            stdlone= max(Dict_mat[layer].keys())
            
            MRCA = True
            continue
        
        hap= list(Dict_mat[layer][-2])
        hap_coord= {}
        
        point_up[layer]= []
        
        Dict_mat[layer + 1]= {   
        }
        
        Quasi= []
        nodes= []
        new_haps= []
        
        keys_get= list(Dict_mat[layer].keys())
        
        for desc in keys_get:
            
            if desc >= 0:
                
                packet= list(Dict_mat[layer][desc])
                packet= np.array(packet)

                pack_above= [x for x in range(packet.shape[0]) if packet[x,1] > 1]
                pack_below= [x for x in range(packet.shape[0]) if packet[x,1] == 1]
                                                
                ## muts that can be removed. Assume mutations happen only once.
                exist= np.array(packet)[:,0]
                exist= np.array(hap)[exist,:]
                single= np.sum(exist,axis= 0)
                single= np.where(single==1)[0]
                ##
                    
                for edan in pack_below:
                    #
                    seq= hap[packet[edan,0]]
                    
                    #print(seq)
                    who= np.where(seq == 1)[0]
                    
                    who= [x for x in who if x in single]
                    
                    if len(who) == 0:
                        if sum(seq) == 0:
                            continue
                        single= np.sum(exist,axis= 0)
                        min_m= min([x for x in single if x != 0])
                        single= np.where(single == min_m)[0]
                        
                        who= np.where(seq == 1)[0]
                        who= [x for x in who if x in single]
                        
                    
                                        
                    for mut in who:
                        
                        tribia= list(seq)
                        tribia= np.array(tribia)
                        tribia[mut]= 0

                        calc= pairwise_distances(np.array(tribia).reshape(1,-1), hap,
                                                        metric='hamming')[0]
                        
                        match= [x for x in range(len(calc)) if calc[x] == 0] 
                        
                        if len(match):
                            #print(match)
                                                        
                            for cl in match:
                                
                                pack_synth= list(Dict_mat[layer][desc])
                                pack_synth= np.array(pack_synth)
                                pack_synth[edan,1] -= 1
                                #print('he')
                                pack_synth= pack_synth[pack_synth[:,1] > 0]
                                
                                if cl in pack_synth[:,0]:
                                    cl_idx= list(pack_synth[:,0]).index(cl)
                                    pack_synth[cl_idx,1] += 0
                                    
                                else:
                                    new_row= np.array([cl,1])
                                    pack_synth= np.vstack((pack_synth,new_row.reshape(1,-1)))
                                
                                #### make function Query existant
                                new_entry= len(Dict_mat[layer + 1])
                                
                                while new_entry in Dict_mat[layer + 1].keys():
                                    new_entry += 1
                                
                                ###
                                path_find= 0 #########
                                pack_tuple= sorted([tuple(x) for x in pack_synth])

                                Query= [pack_tuple == x for x in Quasi]
                                Query= np.array(Query,dtype= int)
                                Query= np.where(Query == 1)[0] ## check if this changes anything

                                if len(Query):
                                    new_entry= nodes[Query[0]]

                                else:
                                    #print(pack_synth)
                                    pack_synth= np.array([list(x) for x in pack_tuple])
                                    Dict_mat[layer + 1][new_entry]= pack_synth
                                    Quasi.append(pack_tuple)
                                    nodes.append(new_entry)
                                ### 

                                point_up[layer].append([desc,new_entry,cl,path_find,mut]) ############
                        
                        else:
                            #
                            
                            if len(new_haps):
                                #
                                calc= pairwise_distances(np.array(tribia).reshape(1,-1), np.array(new_haps),
                                                                                        metric='hamming')[0]
                                
                                match= [x for x in range(len(calc)) if calc[x] == 0]
                                
                                if len(match):
                                    
                                    new_idx= len(hap) + match[0]
                                
                                else:
                                    new_haps.append(tribia)
                                    new_idx= len(hap) #+ len(new_haps) - 1
                            
                            else:
                                new_idx= len(hap)
                                hap.append(tribia)                            
                            #
                            
                            pack_synth= list(Dict_mat[layer][desc])
                            pack_synth.append([new_idx,1]) # pack_synth.append([len(pack_synth),1])
                            pack_synth= np.array(pack_synth)
                            pack_synth[edan,1] -= 1
                            pack_synth= pack_synth[pack_synth[:,1] > 0]
                            
                            #### make function Query existant
                            new_entry= len(Dict_mat[layer + 1])
                            while new_entry in Dict_mat[layer + 1].keys():
                                new_entry += 1
                            
                            ###
                            path_find= 0 #########
                            pack_tuple= sorted([tuple(x) for x in pack_synth])

                            Query= [pack_tuple == x for x in Quasi]
                            Query= np.array(Query,dtype= int)
                            
                            Query= np.where(Query == 1)[0] ## check if this changes anything

                            if len(Query):
                                new_entry = nodes[Query[0]]
                            
                            else:
                                
                                pack_synth= np.array([list(x) for x in pack_tuple])
                                Dict_mat[layer + 1][new_entry]= pack_synth
                                Quasi.append(pack_tuple)
                                nodes.append(new_entry)

                            ####
                            point_up[layer].append([desc,new_entry,new_idx,path_find,mut])
        
        if new_haps:
            
            hap.extend(new_haps)
        
        point_up[layer]= np.array(point_up[layer])
        Dict_mat[layer + 1][-2] = np.array(hap)
        
        layer += 1
    
    t2 = time.time()
    tscale= 's'
    tpass= t2 - t1
    
    if tpass > 600:
        tpass = tpass / 60
        tscale= 'm'
    
    tpass= round(tpass,3)
    if print_time:
    
        print('time elapsed: {} {}'.format(tpass,tscale))
    
    return Dict_mat, point_up

