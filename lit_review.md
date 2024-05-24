## NeRF-LiDAR-cGAN

#### Review
- Interesting sampling / embedding method with kNN

#### Code
- Can visualize demo, but fail to run training (Index Error in dgl library)
- Need to further study dgl library
- Outputs only depth map + Image


## NeRF-LOAM
- NeRF based lidar 

#### Review
- Use Octree's to split scene to axis-aligned voxels, then apply embedding to each vertix
- Then, for each lidar ray, it will be sampled through at place where the ray intersect with voxels
- Neural SDF will then provide SDF field with trilinear interpolation
- Then, lidar sensor pose of each lidar point can be estimated by minimizing SDF error
- Finally, mesh is produced using marching cube method with given SDF network



#### Code
- Error running install.sh due to cuda version mismatch
- however CUDA 10.2 require ubuntu 18.04
- Will seek to ask colleagues for help, or attempt to reinstall ubuntu


## SiLVR

#### Review
- Lidar pior for NeRF, with high quality 3d reconstruction available 
- Apply depth regularization from LiDAR prior:
  $$ L_depth = sum (KL ( Norm(D, sigma) || h(t))) $$
where D is the measured depth

- Apply following surface normal regularization to avoid wavy surface
  $$  $$


#### Code
- Code not available


## Nerfacto

#### Review
- 

#### Code
- Code readily available

## LiDENeRF

#### Code
- Code not yet available



## LiDAR4D

#### Review
- NeRF for LiDAR view synthesis
- State of the art LiDAR view synthesis quality on KITTI 360 data





## 3D LiDAR Recon wiht Probablistic Depth
#### Review
- Uses camera + 16 beam lidar sensor and achieved superior depth reconstruction compare to only using 64 beam lidar sensor
- 

#### Code
- No code available





## Shine Mapping
### Review
- Very important reference for this project
- Author uses OctTree to provide spacial encoding
- For each "ray", ray is sampled at areas close to termination of of laser measurement
- Sampled point is added with trilinear interpolation of encoding values on corners of each octtree embedding

### Model Architecture & Feature Engineering
- Feature encoded with octree with different resoluiton
- Then apply shallow MLP for SDF prediction

### Sampling Strategy
- Sample points along each laser "ray", which can also be over the termination point of the ray

### Loss function Design
- For a sampled point x_i, map a signed distance function onto it, where "0" is at the termination point of the ray
- then, apply sigmoid function to the sign distance function value as said
- This will be the desired value l_i (y for supervised learning)
- On the other hand, the network would predict a signed distance value for given x_i, which would then also be applied a sigmoid function to it, giving us o_i
- Finally, binary cross entropy loss was calulated between o_i and l_i
- Effectively, o_i is the occupancy probability (assume solid after surface)
- Also, use of sigmoid realize soft truncation of signed distance.
- The final loss fo a batch is then:
$$ L_{batch} =  L_{bce} + \lambda_e(|| \frac{df_\theta(x_i)}{dx_i} - 1 ||)^2 + \lambda_rL_r$$ 

- Where L_bce is the cross entropy loss, and the second term is to force function to be SDF
- Finally, L_r term is added so that parameters don't update too much, which can avoid catrosphical forgetting
- With the obtained signed distance field, 3D model can be randered using raymarching.



## VDB fusion
### Review
- Compared to octree, VDB (just a name) is a more effective data structure to store point clouds
- Using point cloud and sensor pose as input, the author can "integrate" all points into one VDB
- Then, for each "ray", near the termination of each ray, we update nearby voxel along the ray
- The voxel to be updated will be updated by weighted projected signed distance value of the ray using centre of each voxel, 
- Finally, the 0 set represents the reconstructed surface
- Technically no machine learning involved


### Code
- Code available in github, and was able to replicate result on KITTI dataset




# Idea

- Model predict 2D surface at each 3d point
- attempt fit fragment of surface to neighbour points
- also punish model for "cutting rays (parellel computation process)

__Concerns__:
- Non-differentiable process
- Long computational time