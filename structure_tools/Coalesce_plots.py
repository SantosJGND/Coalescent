import numpy as np
import itertools as it
import pandas as pd

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot
import scipy

from structure_tools.Coal_probab import *
from structure_tools.Coal_tools import get_sinks

import time
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA

import collections

def recursively_default_dict():
        return collections.defaultdict(recursively_default_dict)


def plot_Ewens(config_complex, range_theta):
    Ncols= 2
    titles= ['AC: {}'.format(''.join(np.array(x,dtype= str))) for x in config_complex]
    print(titles)

    fig_subplots = tools.make_subplots(rows= int(len(titles) / float(Ncols)) + (len(titles) % Ncols > 0), cols=Ncols,
                             subplot_titles=tuple(titles))

    for gp in range(len(titles)):

        pos1= int(float(gp) / Ncols) + 1
        pos2= gp - (pos1-1)*Ncols + 1

        title= titles[gp]


        Ewens_rec= []
        Ewens_ex= []
        there= []
        config_data= config_complex[gp]

        for x in range_theta:

            prob_array= []
            Pin= 1

            probe_rec= Ewens_recurs(config_data,x,prob_array,Pin)
            probe_rec= sum(probe_rec)

            probe_ex= Ewens_exact(config_data,x)

            Ewens_rec.append(probe_rec)
            Ewens_ex.append(probe_ex)
            there.append(x)

        trace1= go.Scatter(
            y= Ewens_rec,
            x= there,
            mode= 'markers',
            name= 'rec'
        )


        trace2= go.Scatter(
            y= Ewens_ex,
            x= there,
            mode= 'markers',
            name= 'exact'
        )

        fig_subplots.append_trace(trace1, pos1, pos2)
        fig_subplots.append_trace(trace2, pos1, pos2)

        fig_subplots['layout']['yaxis' + str(gp + 1)].update(title= 'P')
        fig_subplots['layout']['yaxis' + str(gp + 1)].update(range= [0,.6])
        fig_subplots['layout']['xaxis' + str(gp + 1)].update(title= 'theta')

    layout = go.Layout(
        title= title
    )

    fig= go.Figure(data=fig_subplots, layout=layout)
    
    iplot(fig_subplots)



def plot_rec_InfSites(point_up,root_lib,funk,titles,range_theta,height= 500,width= 900):
    Ncols= 1

    fig_subplots = tools.make_subplots(rows= int(len(titles) / float(Ncols)) + (len(titles) % Ncols > 0), cols=Ncols,
                             subplot_titles=tuple(titles))

    for gp in range(len(titles)):

        pos1= int(float(gp) / Ncols) + 1
        pos2= gp - (pos1-1)*Ncols + 1

        title= titles[gp]


        Inf_sites_est= []
        there= []
        
        runUp_use= funk[gp]
        
        t1 = time.time()
        
        for x in range_theta:
            
            ## run up the tree.
            Browse= runUp_use(point_up,root_lib,layer=0,start=0,Theta= x,prob_vec= [])
            probe_rec= sum(Browse)
            
            Inf_sites_est.append(probe_rec)
            there.append(x)
        
        t2 = time.time()
        tscale= 's'
        tpass= t2 - t1

        if tpass > 600:
            tpass = tpass / 60
            tscale= 'm'

        tpass= round(tpass,3)
        
        trace1= go.Scatter(
            y= Inf_sites_est,
            x= there,
            mode= 'markers',
            name= titles[gp]
        )
        
        fig_subplots.append_trace(trace1, pos1, pos2)
        
        fig_subplots['layout']['yaxis' + str(gp + 1)].update(title= 'P')
        fig_subplots['layout']['yaxis' + str(gp + 1)].update(range= [0,max(Inf_sites_est) + max(Inf_sites_est)/10])
        fig_subplots['layout']['xaxis' + str(gp + 1)].update(title= 'theta - ts {} {}'.format(tpass,tscale))

    layout = go.Layout(
        title= title,
    )

    fig= go.Figure(data=fig_subplots, layout=layout)
    
    fig['layout'].update(height= height,width= width)
    
    
    
    iplot(fig)


def plot_InfSites_mrca(mrcas,point_up,root_lib,range_theta,height= 500,width= 900):
    
    from structure_tools.Coal_tools import tree_descent
    
    Ncols= 1
    titles= [''.join([str(x) for x in y]) for y in mrcas]
    
    fig= []

    for gp in range(len(titles)):
        
        title= titles[gp]

        sink, starters= get_sinks(mrcas[gp],root_lib,point_up)
        
        t1 = time.time()
        if len(starters):

            Inf_sites_est= []
            there= []
        
            for thet in range_theta:
                
                
                node_weigths, paths_reverse = tree_descent(root_lib,point_up,sink,init= starters,Theta= thet)
                
                probe_rec= node_weigths[0][0]

                Inf_sites_est.append(probe_rec)
                there.append(thet)
        
        
            trace1= go.Scatter(
                y= Inf_sites_est,
                x= there,
                mode= 'markers',
                name= titles[gp]
            )

            fig.append(trace1)
    
    
    layout = go.Layout(
        title= title,
        xaxis= dict(
            title= 'Theta'
        ),
        yaxis= dict(
            title= 'P'
        )
    )
    
    iplot(fig)



######
###### gens haps


def plot_InfSites_gens(mrcas,point_up,root_lib,range_theta,Theta= 1,mut_rate= 9e-8,height= 500,width= 900):
    
    from structure_tools.Coal_tools import tree_descent_gen
    
    hap_frame= [[''.join([str(x) for x in z])] for z in mrcas]
    hap_frame= list(range(len(mrcas)))

    hap_frame= pd.DataFrame(hap_frame,columns= ['hap_id'])
    hap_frame["hap"]= [''.join([str(x) for x in z]) for z in mrcas]
    
    Ncols= 1
    titles= [''.join([str(x) for x in y]) for y in mrcas]
    
    fig= []
    vals= []

    for gp in range(len(titles)):
        
        title= titles[gp]

        sink, starters= get_sinks(mrcas[gp],root_lib,point_up)
        
        t1 = time.time()
        if len(starters):
        
            #node_weigths, paths_reverse = tree_descent(root_lib,point_up,sink,init= starters,Theta= thet)

            #probe_rec= node_weigths[0][0]

            node_weigths, paths_reverse, node_bins, paths_vector = tree_descent_gen(root_lib,point_up,sink,Theta= Theta,mu= mut_rate)
            
            paths_vector= paths_reverse[0][0]
            average_gen= np.mean(paths_vector)
            
            vals.append(average_gen)
    
    
    fig = [go.Bar(
        x= ['hap: {}'.format(x) for x in range(len(titles))],
        y= vals
    )]
            
    
    layout = go.Layout(
        title= 'gens until first hap appearence',
        barmode='group',
        xaxis= dict(
            title= 'hap'
        ),
        yaxis= dict(
            title= 'Gen'
        )
    )
    
    Figure= go.Figure(data= fig, layout= layout)
    
    hap_frame['t']= [round(c,3) for c in vals]
    
    return hap_frame, Figure


############
############



def plot_phyl_net(data_phyl,leaves,node_list,edges,nodes_as_seqs= True,root= True):
    import networkx as nx

    G=nx.Graph()

    G.add_nodes_from(node_list)
    G.add_edges_from(edges)

    pos= nx.fruchterman_reingold_layout(G)

    ###
    ### labels 
    for nd in node_list:
        if nd not in leaves.keys():
            leaves[nd]= []

    if nodes_as_seqs:
        labels= []
        for nd in node_list:
            if len(leaves[nd]):
                seqs= [data_phyl[x] for x in leaves[nd]]
                seqs= [''.join([str(x) for x in z]) for z in seqs]
                seqs= '\n'.join(seqs)
                labels.append(seqs)
            else:
                labels.append('')

    else:

        labels= [''.join([str(x) for x in leaves[z]]) for z in node_list]
    
    ### colors
    colz= ['rgb(0,0,205)']*len(labels)
    if root:
        where_root= node_list.index(-1)
        colz[where_root]= 'rgb(240,0,0)'
        labels[where_root] = 'root: ' + labels[where_root]
    
    ##
    Xn=[pos[k][0] for k in pos.keys()]
    Yn=[pos[k][1] for k in pos.keys()]

    trace_nodes=dict(type='scatter',
                     x=Xn, 
                     y=Yn,
                     mode='markers',
                     marker=dict(size=28, color=colz),
                     text=labels,
                     hoverinfo='text')

    Xe=[]
    Ye=[]

    for e in G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])

    trace_edges=dict(type='scatter',
                 mode='lines',
                 x=Xe,
                 y=Ye,
                 line=dict(width=1, color='rgb(25,25,25)'),
                 hoverinfo='none' 
                )

    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='' 
              )
    layout=dict(title= 'Gene graph',  
                font= dict(family='Balto'),
                width=600,
                height=600,
                autosize=False,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(
                l=40,
                r=40,
                b=85,
                t=100,
                pad=0,

        ),
        hovermode='closest',
        plot_bgcolor='#efecea', #set background color            
        )

    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )
    
    layout=dict(title= 'My Graph',  
                font= dict(family='Balto'),
                width=600,
                height=600,
                autosize=False,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(
                l=40,
                r=40,
                b=85,
                t=100,
                pad=0,

        ),
        hovermode='closest',
        plot_bgcolor='#efecea', #set background color            
        )


    fig = dict(data=[trace_edges, trace_nodes], layout=layout)
    iplot(fig)


###################################
###################################


def get_ori_graph(root_lib,edges,node_list,leaves,present= True,
                                            nodes_as_seqs= True,
                                            root= True):
    
    import networkx as nx
    
    str_data= [''.join([str(x) for x in z]) for z in root_lib[0][-2]]

    ##
    node_list= sorted(list(set(it.chain(*edges))))

    G=nx.Graph()

    G.add_nodes_from(node_list)
    G.add_edges_from(edges)

    pos= nx.fruchterman_reingold_layout(G)

    ###
    ### labels 
    for nd in node_list:
        if nd not in leaves.keys():
            leaves[nd]= []

    if nodes_as_seqs:
        labels= []
        for nd in node_list:
            if len(leaves[nd]):
                seqs= ''.join([str(x) for x in leaves[nd]])
                labels.append(seqs)
            else:
                labels.append('')

    else:
        labels= [''.join([str(x) for x in leaves[z]]) for z in node_list]


    ### colors
    colz= ['rgb(186,85,211)']*len(labels)

    if present:
        list_p= [x for x in range(len(node_list)) if ''.join([str(g) for g in leaves[node_list[x]]]) in str_data]
        print(list_p)
        for h in list_p:
            colz[h]= 'rgb(0,0,205)'

    if root:
        where_root= node_list.index(-1)
        colz[where_root]= 'rgb(240,0,0)'
        labels[where_root] = 'root: ' + labels[where_root]

    ##
    Xn=[pos[k][0] for k in pos.keys()]
    Yn=[pos[k][1] for k in pos.keys()]

    trace_nodes=dict(type='scatter',
                     x=Xn, 
                     y=Yn,
                     mode='markers',
                     marker=dict(size=28, color=colz),
                     text=labels,
                     hoverinfo='text')

    Xe=[]
    Ye=[]

    for e in G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])

    trace_edges=dict(type='scatter',
                 mode='lines',
                 x=Xe,
                 y=Ye,
                 line=dict(width=1, color='rgb(25,25,25)'),
                 hoverinfo='none' 
                )

    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='' 
              )
    layout=dict(title= 'Full ancestry graph',  
                font= dict(family='Balto'),
                width=600,
                height=600,
                autosize=False,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(
                l=40,
                r=40,
                b=85,
                t=100,
                pad=0,

        ),
        hovermode='closest',
        plot_bgcolor='#efecea', #set background color            
        )

    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )

    layout=dict(title= 'My Graph',  
                font= dict(family='Balto'),
                width=600,
                height=600,
                autosize=False,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(
                l=40,
                r=40,
                b=85,
                t=100,
                pad=0,

        ),
        hovermode='closest',
        plot_bgcolor='#efecea', #set background color            
        )


    fig = dict(data=[trace_edges, trace_nodes], layout=layout)
    iplot(fig)


######################
######################


def theta_PCAms_plot(data_combs,Z,N_samp= 50,n_comp= 4):
    ## Perform PCA
    n_comp= 4
    pca = PCA(n_components=n_comp, whiten=False,svd_solver='randomized')
    ##

    feats_combs= pca.fit_transform(data_combs)

    fig_pca_combs= [go.Scatter3d(
        x= feats_combs[:,0],
        y= feats_combs[:,1],
        z= feats_combs[:,2],
        mode= 'markers',
        marker= dict(
            color= Z.reshape(1,-1)[0],
            colorscale= 'Viridis'
        )
        )
    ]

    ###
    ###
    Z_chose= list(Z.reshape(1,-1)[0])
    Z_chose= np.argsort(Z_chose)

    Z_chose= Z_chose[(len(Z_chose) - 15):]
    #Z_chose= [x for x in range(len(Z_vec)) if Z_vec[x] >= 1]

    Z_high= feats_combs[Z_chose]

    print(Z_high.shape)
    params = {'bandwidth': np.linspace(0.1, 2, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
    grid.fit(Z_high)

    kde = grid.best_estimator_
    new_data = kde.sample(N_samp, random_state=0)

    fig_pca_combs.append(go.Scatter3d(
        x= new_data[:,0],
        y= new_data[:,1],
        z= new_data[:,2],
        mode= 'markers'))


    layout= go.Layout(
            scene = dict(
            xaxis = dict(
                 backgroundcolor="rgb(72,61,139)",
                 gridcolor="rgb(255, 255, 255)",
                 showbackground=False,
                 showgrid= False,
                 zerolinecolor="rgb(255, 255, 255)",),
            yaxis = dict(
                backgroundcolor="rgb(72,61,139)",
                gridcolor="rgb(255, 255, 255)",
                showbackground=False,
                showgrid= False,
                nticks=0,
                zerolinecolor="rgb(255, 255, 255)"),
            zaxis = dict(
                backgroundcolor="rgb(72,61,139)",
                gridcolor="rgb(255, 255, 255)",
                showbackground=False,
                showgrid= False,
                zerolinecolor="rgb(255, 255, 255)",),)
    )

    Figure= go.Figure(data= fig_pca_combs,layout= layout)
    
    return Figure, new_data, feats_combs


### random search summary
### 


def PCA_sumplot(Z,zprime,Theta_record,pca_obj,Ncols= 2,PC_select= 2,height= 600, width= 1000):
    titles= ['probab','Ave. PC coordinates among kde sampled theta vectors','loadings of PC {}'.format(PC_select + 1)]

    fig_subplots = tools.make_subplots(rows= int(len(titles) / float(Ncols)) + (len(titles) % Ncols > 0), cols=Ncols,
                             subplot_titles=tuple(titles))
    for gp in range(len(titles)):
        pos1= int(float(gp) / Ncols) + 1
        pos2= gp - (pos1-1)*Ncols + 1

        title= titles[gp]

        if gp== 0:
            zprime= Z[Z_chose]
            bandwidth = estimate_bandwidth(zprime, quantile=0.2, n_samples=500)

            X_plot = np.linspace(-2, 8, 100)[:, np.newaxis]

            kde_plot = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(zprime)
            log_dens = kde_plot.score_samples(X_plot)

            trace= go.Scatter(x=X_plot[:, 0], y=np.exp(log_dens),
                                        mode='lines', fill='tozeroy',
                                        line=dict(color='red', width=2))

            fig_subplots.append_trace(trace, pos1, pos2)
            fig_subplots.append_trace(fig_dens_I[0], pos1, pos2)

        if gp == 1:

            trace= go.Bar(
                        x=['PC {}'.format(x + 1) for x in range(new_data.shape[1])],
                        y=feat_sum,
                        marker= dict(color= 'rgb(0,0,205)')
                )
            fig_subplots.append_trace(trace, pos1, pos2)

            fig_subplots['layout']['yaxis' + str(gp + 1)].update(title='mean')

        if gp == 2:

            times_data= [list(Theta_record[x]['comb'][:,0]) for x in Theta_record.keys()]
            times_data= np.array(times_data)
            times_av= np.mean(times_data,axis= 0)
            times_av= [int(x) for x in times_av / 1000]
            times_av= ['{}k y'.format(x) for x in times_av]

            Xcomps= pca_obj.components_ 

            trace= go.Bar(
                        x=times_av,
                        y=Xcomps[PC_select,:],
                        marker= dict(color= 'rgb(0,0,205)')
                )

            fig_subplots.append_trace(trace, pos1, pos2)

            fig_subplots['layout']['yaxis' + str(gp + 1)].update(title='eigen value')

    fig_subplots['layout'].update(height= height, width= width) 

    layout = go.Layout(
        title= title
    )

    fig= go.Figure(data=fig_subplots, layout=layout)
    iplot(fig_subplots)



#### plot theta in time
####

