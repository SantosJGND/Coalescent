import numpy as np
import itertools as it
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot
import scipy

from structure_tools.Coal_probab import *

import time

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
