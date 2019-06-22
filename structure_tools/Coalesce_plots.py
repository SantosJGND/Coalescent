import numpy as np
import itertools as it
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot
import scipy

from structure_tools.Coal_probab import *
from structure_tools.Coal_tools import get_sinks

import time


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






def plot_phyl_net(data_phyl,leaves,node_list,edges,nodes_as_seqs= True,root= True):
    import networkx

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



def get_ori_graph(root_lib,edges,node_list,present= True,
                                            nodes_as_seqs= True,
                                            root= True):
    
    import networkx
    
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
