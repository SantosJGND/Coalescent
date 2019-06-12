import numpy as np
import itertools as it
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot
import scipy

from structure_tools.Coal_probab import *

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