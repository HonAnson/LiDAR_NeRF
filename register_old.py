import os
import numpy as np
from einops import rearrange, repeat
import pandas as pd         # for loadData()
import open3d as o3d        # for getting point cloud register

def cart2sph(xyz):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(list(XsqPlusYsq + z**2))
    elev = np.arctan2(list(z), np.sqrt(list(XsqPlusYsq)))
    pan = np.arctan2(list(y), list(x))
    output = np.array([r, elev, pan])
    return rearrange(output, 'a b -> b a') #take transpose

def sph2cart(ang):
    ele = ang[:,0]
    pan = ang[:,1]
    x = np.cos(ele)*np.cos(pan)
    y = np.cos(ele)*np.sin(pan)
    z = np.sin(ele)
    output = np.array([x,y,z])
    return rearrange(output, 'a b -> b a') #take transpose


def loadData(dataset_path):
    # List all files in the specified path, ignoring directories
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    files.sort()
    # read the files
    points_xyz = []
    for s in files:
        path = dataset_path + s
        df = pd.read_csv(path)
        a = df.to_numpy()
        points_xyz.append(a[:,8:11])
    return points_xyz


def getTransformation(data):
    """ 
    Accept input of list of numpy array of n*3 size, 
    return list of 4*4 numpy array of transformation matrix
    each are transformation needed from each frame to point cloud in fame 0
    """
    # convert numpy array 
    point_clouds = []
    for d in data:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(d)
        point_clouds.append(pcd)
    threshold = 2.0 
    trans_init = np.eye(4)  # Initial transformation matrix
    transformations = []

    # the following may take a while
    for i in range(len(point_clouds)):
        source_pcd = point_clouds[i]
        target_pcd = point_clouds[0]
        reg_p2p = o3d.pipelines.registration.registration_icp(
                    source_pcd, 
                    target_pcd, 
                    threshold, 
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transformations.append(reg_p2p.transformation)
    return transformations
    
def prepareData(points_xyz):
    # get transformations
    # transformations = getTransformation(points_xyz)
    # register all points onto same global coordinte (global coordinate align with frame 0 coordinate)
    # points_reg_xyz = []
    # for i, points in enumerate(points_xyz):
    #     ones = np.ones((len(points), 1))
    #     homo_points = np.hstack((points, ones))
    #     # apply transformation
    #     t = transformations[i]
    #     t = rearrange(t, 'a b -> b a')
    #     reg_points = homo_points@t
    #     reg_points = reg_points / rearrange(reg_points[:,3], 'a -> a 1')
    #     reg_points = reg_points[:,0:3]
    #     points_reg_xyz.append(reg_points)

    # create a list of origins 
    # centres = [(t@np.array([[0],[0],[0],[1]]))[0:3,0] for t in transformations]
    centres = np.array([
        [0,0,0],
        [1.1854,0,0],
        [2.1397,0,0],
        [3.5695,0,0],
        [4.4137,0,0]
    ])    

    # get the angular direction and distance of rays    
    points_sph = []
    # for i, points in enumerate(points_reg_xyz):
    for i, points in enumerate(points_xyz):
        # relative_loc = points - centres[i]
        relative_loc = points
        points_sph.append(cart2sph(relative_loc))

    # tile sensor centre 
    centres_tiled = []
    for i, centre in enumerate(centres):
        l = len(points_sph[i])
        temp = np.tile(centre, (l, 1))
        centres_tiled.append(temp)
    
    # stack everything into a matrix of size n * 6
    # where n is number of points, 6 corrsponds to:
    # distance, elevation, pan, x of camera, y of camera, z of camera
    # stack the points into one big matrix
    stacked = []
    for i in range(len(points_sph)):
        temp = np.hstack((points_sph[i], centres_tiled[i]))
        stacked.append(temp)

    dataset = np.array([])
    for i in range(len(stacked)):
        if i == 0:
            dataset = stacked[i]
        else:
            dataset = np.vstack((dataset, stacked[i]))

    # Mid pass filter, for distance value between 2 and 50 meter
    mask1 = dataset[:,0] > 2
    dataset = dataset[mask1]
    mask2 = dataset[:,0] < 50
    dataset = dataset[mask2]
    np.random.shuffle(dataset)      # shuffle for good practice
    return dataset


if __name__ == "__main__":
    path = r'datasets/testing1/'
    a = loadData(path)
    data = prepareData(a)
    r = data[:,0]
    ang = data[:,1:3]
    o = data = data[:,3:6]

    dir = sph2cart(ang)
    r = rearrange(r, 'a -> a 1')
    pos_np = dir*r + o
    to_save = np.hstack((r, ang, o))
    np.save('manual_register', to_save)

    ### Save to csv for visualization
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    df_temp = df_temp.head(pos_np.shape[0])
    df_temp['X'] = pos_np[:,0]
    df_temp['Y'] = pos_np[:,1]
    df_temp['Z'] = pos_np[:,2]
    df_temp.to_csv('local/visualize/register_check3.csv', index=False)









