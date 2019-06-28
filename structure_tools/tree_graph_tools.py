
import collections

def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)


import numpy as np 
import itertools as it 


def Gus_phylo_test(hap_array):
    
    ## get cols
    Tree= np.array(hap_array,dtype= str).T
    
    ### get binary of cols
    binary_t= [''.join(x) for x in Tree]
    binary_t= [int(x,2) for x in binary_t]
    
    ### get sort of col binary
    bin_sort= np.argsort(binary_t)[::-1]
    
    ### remove duplicates
    Tree_str= [''.join(x) for x in Tree]
    
    Mp= [Tree_str[x] for x in bin_sort if binary_t[x]]
    
    dup_try= [0,*[x for x in range(1,len(Mp)) if Mp[x] != Mp[x - 1]]]
    
    Mp= [Mp[x] for x in dup_try]
    
    ## get Mprime to original array col index for phyl construction later.
    Mp_similarity= {
        z: [x for x in range(len(Tree_str)) if Tree_str[x] == Mp[z]] for z in range(len(Mp))
    }
    
    Mp= [list(x) for x in Mp]
    
    ### get haps back cols sorted, nulls removed.
    Mp= np.array(Mp,dtype= int).T

    ## get all positive cells.
    valid_c= np.where(Mp == 1)
    where_one= [tuple([valid_c[0][x],valid_c[1][x]]) for x in range(len(valid_c[0]))]
    
    ## cells by row
    where_lib= [c[0] for c in where_one]
    where_lib= {
        z:[where_one[x] for x in range(len(where_lib)) if where_lib[x] == z] for z in list(set(where_lib))
    }
    
    # cells in the same row but previous col for every cell
    where_prime= [
        [z[1] for z in where_lib[cel[0]] if z[1] < cel[1]] for cel in where_one
    ]
    
    ## respective cols L(i,j)
    where_prime= [[[-1],x][int(len(x) > 0)] for x in where_prime]
    where_prime= [max(x) for x in where_prime]

    ## cells by col
    col_lib= [c[1] for c in where_one]
    
    ## largest smaller index by col
    col_lib= {
        z:[where_prime[x] for x in range(len(col_lib)) if col_lib[x] == z] for z in list(set(col_lib))
    }
    
    ## check how many different previous to larger col indeces by cell
    col_lib= {z:list(set(g)) for z,g in col_lib.items()}
    
    discovery= np.array([len(x) for x in col_lib.values()])
    
    ## According to Gusfield:
    '''
    - Check whether L(i,J) = Lo') for every cell (ij) E 0. If so. then M has
    a phylogenetic tree: otherwise. M does not have one.
    '''
    
    ## i'm interpreting this as: all previous to last col indeces must be same by col index. 
    
    discovery= np.where(discovery > 1)[0]
    discovery= len(discovery) == 0
    return discovery, Mp, col_lib, Mp_similarity


##### get phylogeny. Gusfield 1991

def Gus_get_phylo(Mp,col_lib,Mp_similarity):
    node_edges= recursively_default_dict()
    nodes_all=[]
    
    node_edges[-1]= {}

    root= {}

    tree_nodes= recursively_default_dict()
    
    leaves= recursively_default_dict()

    for col,L in col_lib.items():
        
        node_edges[L[0]][col]= Mp_similarity[col]
    
    
    for ri in range(Mp.shape[0]):
        row= Mp[ri]

        ci= np.where(row == 1)[0]

        if len(ci):
            ci= max(ci)
            edges= [tuple([z,ci]) for z in node_edges.keys() if ci in node_edges[z]]

            for ed in edges:
                leaves[ed[1]][ri]= 1

        else:
            leaves[-1][ri]= 1

    leaves= {
        z: list(leaves[z].keys()) for z in leaves
    }

    ## because this format might prove more useful later
    edges= [[(x,z) for z in node_edges[x]] for x in node_edges.keys()]
    edges= list(it.chain(*edges))

    return node_edges, leaves, edges


#####################################################
################# NETWORKS ##########################


def tree_descent_net(root_lib,point_up,sink,init= [0]):
    from sklearn.metrics import pairwise_distances
    
    for layer in list(range(1,sink + 1))[::-1]:
        
        where_to= point_up[layer - 1]
        
        if layer == sink:
            starters= init
        else:
            starters= list(set(where_to[:,1]))

        point_up_house= where_to[where_to[:,3] == 0]

        if layer == sink:
            
            AC= root_lib[sink][0][0]
            
            leaves= {
                -1: root_lib[sink][-2][AC[0]]
            }
            
            nodes= {
                -1: []
            }
            
            node_code= {
                AC[0]: -1
            }
            
            edges= []
        
        
        for row in range(point_up_house.shape[0]):
            line= point_up_house[row,:]
            
            who_app= [x for x in root_lib[layer - 1][line[0]][:,0] if x not in root_lib[layer][line[1]][:,0]][0]
            
            if who_app not in node_code.keys():
                node_code[who_app]= len(node_code) - 1
                
                nodes[node_code[who_app]]= []
                
                leaves[node_code[who_app]]= root_lib[layer][-2][who_app]
            
            mut= point_up_house[row,4]
            
            who_seq= list(root_lib[layer - 1][-2][who_app])
            
            who_seq[mut]= 0
            
            dists= pairwise_distances(np.array(who_seq).reshape(1,-1), root_lib[layer][-2],
                                                metric='manhattan')[0]
            
            which= np.where(dists==0)[0]
            
            for poss in which:
                
                if poss != who_app:
                    
                    if poss not in node_code.keys():

                        node_code[poss]= len(node_code) - 1

                        nodes[node_code[poss]]= []

                        leaves[node_code[poss]]= root_lib[layer][-2][poss]
                    
                    code_dn= node_code[poss]

                    new_edge= (code_dn,node_code[who_app])
                    
                    if new_edge not in edges:
                        edges.append(new_edge)

                        nodes[new_edge[0]].append(new_edge[1])

    return nodes, edges, leaves, node_code



