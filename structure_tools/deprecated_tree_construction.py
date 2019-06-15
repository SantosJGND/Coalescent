


### why complicate?

import time
from sklearn.metrics import pairwise_distances


def Inf_sites_autocount(Dict_mat,point_up,layer_range= 10, prob_vec= [],Theta= 1,sub_sample= 0, poppit= False):
    
    paths= recursively_default_dict()
    paths_where= recursively_default_dict()
    path_weights= recursively_default_dict()
    
    edge_weights= recursively_default_dict()
    t1 = time.time()
   
    MRCA= False
    
    layer= 0
    
    
    for layer in range(layer_range):
        
        if MRCA:
            continue
            
        #print('layer: {}; len: {}'.format(layer,len(Dict_mat[layer])-1))
        
        if len(Dict_mat[layer]) == 2:
            stdlone= max(Dict_mat[layer].keys())
            if sum(Dict_mat[layer][stdlone][:,1]) == 1:
                MRCA = True
                continue

        if poppit:
            if layer > 1:
                Dict_mat.pop(layer - 1)
            
        hap= list(Dict_mat[layer][-2])
        hap_coord= {}
        
        point_up[layer]= []
        
        Dict_mat[layer + 1]= {   
        }
                
        Quasi= []
        nodes= []
        new_haps= []
        
        keys_get= list(Dict_mat[layer].keys())
        
        if sub_sample:
            keys_get= np.random.choice(keys_get,sub_sample)
        
        for desc in keys_get:
                
            if desc >= 0:
                
                point_up_house= []
                packet= list(Dict_mat[layer][desc])
                packet= np.array(packet)
                
                ###
                nsamp= sum(packet[:,1])

                mut_prob= prob_mut(Theta,nsamp)
                coal_prob= prob_coal(Theta,nsamp)
                
                prob_vec= [mut_prob,coal_prob]
                ###

                pack_above= [x for x in range(packet.shape[0]) if packet[x,1] > 1]
                pack_below= [x for x in range(packet.shape[0]) if packet[x,1] == 1]
                
                new_entries= np.array(list(range(len(pack_above)))) + len(Dict_mat[layer + 1])
                
                who_loses= []
                
                ### Coalescence
                for z in range(len(pack_above)):
                    
                    who_loses.append(packet[pack_above[z],0])
                    
                    pack_synth= list(packet)
                    pack_synth= np.array(pack_synth)

                    pack_synth[pack_above[z],1] -= 1
                    
                    pack_tuple= sorted([tuple(x) for x in pack_synth])
                    
                    Query= [pack_tuple == x for x in Quasi]
                    Query= np.array(Query,dtype= int)
                    Query= np.where(Query == 1)[0] ## check if this changes anything
                    
                    if len(Query):
                        new_entries[z] = nodes[Query[0]]
                        
                    else:
                        pack_synth= np.array([list(x) for x in pack_tuple])
                        
                        pack_synth= pack_synth[pack_synth[:,1] > 0]
                        Dict_mat[layer + 1][new_entries[z]]= pack_synth
                        Quasi.append(pack_tuple)
                        nodes.append(new_entries[z])
                                
                packet_mob= packet[pack_above,:]
                
                packet_mob[:,1] = 1
                
                packet_mob= np.hstack((np.zeros((packet_mob.shape[0], 1), dtype=packet_mob.dtype),packet_mob))
                packet_mob= np.hstack((packet_mob,np.zeros((packet_mob.shape[0], 1), dtype=packet_mob.dtype)))
                packet_mob[:,3] = -1 #######
                packet_mob[:,0]= new_entries
                packet_mob= np.hstack((np.zeros((packet_mob.shape[0], 1), dtype=packet_mob.dtype),packet_mob))
                packet_mob[:,0]= desc
                
                for y in packet_mob:
                    point_up_house.append(y)
                
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
                        continue
                    
                    #
                                        
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
                                pack_synth= pack_synth[pack_synth[:,1] > 0]
                                
                                if cl in pack_synth[:,0]:
                                    cl_idx= list(pack_synth[:,0]).index(cl)
                                    pack_synth[cl_idx,1] += 1
                                    
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
                                Query= np.where(Query == 1)[0] ## 
                                
                                if len(Query):
                                    new_entry= nodes[Query[0]]

                                else:
                                    #print(pack_synth)
                                    pack_synth= np.array([list(x) for x in pack_tuple])
                                    Dict_mat[layer + 1][new_entry]= pack_synth
                                    Quasi.append(pack_tuple)
                                    nodes.append(new_entry)
                                ### 

                                point_up_house.append([desc,new_entry,cl,path_find,mut]) ############
                        
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
                                    new_idx= len(hap) + len(new_haps) - 1
                            
                            else:
                                new_haps.append(tribia)
                                new_idx= len(hap)
                            
                            #
                            pack_synth= list(Dict_mat[layer][desc])
                            pack_synth.append([new_idx,1]) # 
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
                            Query= np.where(Query == 1)[0] ## 

                            if len(Query):
                                new_entry = nodes[Query[0]]

                            else:
                                
                                pack_synth= np.array([list(x) for x in pack_tuple])
                                Dict_mat[layer + 1][new_entry]= pack_synth
                                Quasi.append(pack_tuple)
                                nodes.append(new_entry)
                            
                            ####
                            point_up_house.append([desc,new_entry,new_idx,path_find,mut])
                
                point_up[layer].extend(point_up_house)
                point_up_house= np.array(point_up_house)
                
                if not len(paths):
                    
                    paths= {
                        0: 1
                    }
                    
                    paths_where[layer][desc]= [1]
                
                #print('###')
                #print(paths_where)
                
                for row in range(point_up_house.shape[0]):
                    pack= list(paths_where[layer][desc])
                                        
                    going= point_up_house[row,1]
                    node_next= np.array(Dict_mat[layer + 1][going])
                    
                    who_lost= point_up_house[row,2] # hap set that originates the mutation / coalescent event
                    hap_split= node_next[node_next[:,0] == who_lost] # hap set row
                    prob_split= (hap_split[0,1]) / sum(node_next[:,1])
                    
                    if row > 0:
                        new_entry= len(paths)
                        while new_entry in paths.keys():
                            new_entry += 1
                    
                    #print('layer: {};desc: {}; next: {}; mut: {}; prop: {}; tog: {}'.format(layer,
                    #                                                               desc,going,
                    #                                                               round(prob_vec[point_up_house[row,3]],3),
                    #                                                               prob_split,
                    #                                                               round(prob_split * prob_vec[point_up_house[row,3]],3)))
                    
                    pack= [x * prob_vec[point_up_house[row,3]] * prob_split for x in pack]

                    if going not in paths_where[layer + 1].keys():
                        paths_where[layer + 1][going]= pack
                    
                    else:
                        paths_where[layer + 1][going].extend(pack)
                        
        
        if new_haps:
            
            hap.extend(new_haps)
        
        if len(edge_weights) == 0:
            edge_weights[0][0] = 1
            
        for desc in paths_where[layer + 1].keys():
            edge_weights[layer + 1][desc]= sum(paths_where[layer + 1][desc])
        
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
        
    return Dict_mat, point_up, point_dn, edge_weights, paths_where




def tree_construct(bogus1,bogus2,layer=0,start=0,Theta= 1,prob_vec= []):
    
    point_up= recursively_default_dict()
    
    root_lib, point_up, node_weigths, paths = Inf_sites_autocount(Dict_mat,point_up,
                                                                 layer_range= 10,Theta=Theta,sub_sample= 0,poppit= False)
    
    sink= max(paths.keys())
    
    if 0 not in paths[sink].keys():
        while 0 not in paths[sink].keys():
            sink -= 1
    
    return paths[sink][0]



