import open3d
import numpy 
import torch
from train import LiDAR_NeRF
from einops import repeat, rearrange
from numpy import cos, sin, array
import pandas as pd

def sph2cart(ang):
    ele = ang[:,0]
    pan = ang[:,1]
    x = cos(ele)*cos(pan)
    y = cos(ele)*sin(pan)
    z = sin(ele)
    output = array([x,y,z])
    return rearrange(output, 'a b -> b a') #take transpose


def visualize(model_path, output_path):
    """ Visualize reconstruction from model and position"""
    #### Load the model and try to "visualize" the model's datapoints
    model_evel = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=15, device = 'cpu')
    model_evel.load_state_dict(torch.load(model_path))
    model_evel.eval(); # Set the model to inference mode

    ### Render some structured pointcloud for evaluation
    with torch.no_grad():
        dist = 0.05 # initial distanc for visualization
        pos = torch.zeros((100000,3))
        ele = torch.linspace(-0.34, 0.3, 100)
        pan = torch.linspace(-3.14, 3.14, 1000)
        ele_tiled = repeat(ele, 'n -> (r n) 1', r = 1000)
        pan_tiled = repeat(pan, 'n -> (n r) 1', r = 100)
        ang = torch.cat((ele_tiled, pan_tiled), dim=1)

        # direction for each "point" from camera centre
        directions = torch.tensor(sph2cart(array(ang)))

        for i in range(100):
            output2 = model_evel(pos, ang)
            temp = torch.sign(output2)
            pos += directions * dist * temp
            print(f'visualizing... ({i}/100)')

    ### Save to csv for visualization
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    df_temp = df_temp.head(100000)
    pos_np = pos.numpy()
    df_temp['X'] = pos_np[:,0]
    df_temp['Y'] = pos_np[:,1]
    df_temp['Z'] = pos_np[:,2]
    df_temp.to_csv(output_path, index=False)
    return


if __name__ == "__main__":
    model_path = r'local/models/version4_trial0.pth'
    output_path = r'local/visualize/visualize.csv'
    visualize(model_path,output_path)







