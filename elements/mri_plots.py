import numpy as np
import pandas as pd
import plotly.express as px

def plot_img(data, plane):

    img_rgb = data[plane]
    
    fig = px.imshow(
        img = img_rgb, 
        binary_string=True
        )
    
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'showlegend': False,
        'xaxis':{
            'visible': False
        },
        'yaxis':{
            'visible': False
        },
        })

    return fig

if __name__ == '__main__':

    """
    Function to plot a plane of the MRI images
    """

    case = '0000'
    mri_sagittal = np.load('data/train/sagittal/0000.npy')
