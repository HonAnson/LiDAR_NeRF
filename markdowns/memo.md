### Livox Data CSV

- X, Y, Z, are location of the point, with respect to the camera centre
- No time information provided
- No



### Trial 0
- Used spherical harmonics as encoder for view direction
- Combined measurement from 5 frames into 1 set, assuming linear translation between frames only
- Model architecture with 8 layers of 256 fully connectted neurons
- Tried L2 and L1 loss
- Did not generalize well


### Trial 1
- Use angular encoding same as nerf paper
- Loss function same as SHINE mapping's sigmoid mapping design
- Used sampled position and view direction for each ray as input
- 8 layers of 512 fully connected neuron, with skip connection at between 4th and 5th later
- Embedding dimension was 10 for both position and view direction
- Sampling strategy of uniform sampling + pertebration w/ 100 bins from 3m distance to 75m from camera centre
- Achieved Low loss function value of approximately 0.04, but fail to achieve anything reasonable upon visualization
- Suspect reason due to majority of sampled position are belong positive or negative side of SDF, causing class imbalancing, and thus collaps of model
- Suspect a bug in calculation of loss function
 
### Trial 4
- fixed indexing problem in sampling, which appears to be the cause of previous bug
- Loss value converges at around 0.25 after 1 epoch, relatively high loss
- Currently the sampling method samples mainly "on the ray", will try with a more balancd class in next trial
- if next trial fails, will look into nerf in the wild, or other approach

### Trial 5
- loss convers around 0.2 after 1 epoch
- Able to reconstruct the floor  
- rough reconstruction of vehicles around
- Poor reconstruction to structures
- maybe consider space wrapping?

### Trial 7
- tunned hyperparameter of angle embedding dimension to 15
- No significant improvement could be seen
- Plan: do more literature review
- explore oct-tree embedding instead of positional embedding instead (allow better representation efficiency)
- explore upsampling for small angle 

### Trial 8
- updated sampling strategy so that denser points are sampled around points in point cloud


## Version 2
### Trial 2
- Utilized upsampling
- Slightly better reconstruction overall, but fail to reconstruct nearby structure
- Still amny noises around

### Trial 4
- implemented ICP for frame registration, no huge difference, but does reduce noise
- next step will be to change sampling method

### Trial 5
- tried a more "proper" sampling strategy, where points close to surface are densely sampled

### Trial 8 and 9
- tried smaller scale scene
- very poor reconstruction quality achieved

### Trial 10
- using manual registered points, with upsampling
- About 280 second per iteration


### Trial 11
- using manual register points, without upsampling


### Trial 12
- Used MSE loss instead of sigmoid and BCE loss
- Achieved similar result with sigmoid and BCE loss

## Version 
- Trying to use space wrapping 
- In particular, I first try to use polar angle coordinate to express view point instead of xyz coordinate
- 
### Trial 1
- attempt to use polar coordinate for training, doesn't appear to improve loss value
- Reconstruction quality lower than using cartisian coordinate


### Trial 2
- Applied wrapping with radius 10 around [0,0,0], hopefully would help
- Also used manual registering of the data, whcich shall improve reconstruction quality


## Version 4
- Using object centered data, that is known to be more suitable for nerf models
- Data are collected manually

### Trial 0
- Tried using same model as previou part, but using object centered data
- Didn't achieve any good result compared to previous method

### Trial 1
- trained model on data box_plant2.npy
- Using same model architecture
- Fixed logic bug in sampling
- Using updated sampling method (inverse sigmoid from univorm distribution)
- Unable to judge well reconstruction result due to lack of visualization function
- Trained model with 8 epoch, batchsize 1024

### Trial 2
- Settings same as v4trial1, but trained on round plant data, which has better registeration
- Trained model with 16 epoch, used around 12 hours on nvidia 3050gpu, batchsize 1024

### Trial 3
- Same setting as previou, using building data
this time using MSE loss

### TODO
* check whether upSampling() is working properly, where we are always using the smaller angle
* Work on visualization
* Figure out how to use HPC in campus
* Work on adding wrapping to the model
* Try to skew the sampling
* Scale all scenes such that they are within certian range (maybe -10 to 10)
* 


### Note to self:
1. update design of loss function to: either include implciit LOD or enforce projected TSDF instead of the current SDF
  
2. Try Ekironal Loss

3. try to use TSDF instead of the current weird project sdf range

4. Download other dataset and give it a go


- According to paper `DS-NeRF`, they concluded that the ideal ray termination distribution approaches a δ impulse function. Where the variance of `Occupancy` approach 0 In the context of LiDAR measurement, the delta impulse function is actually known. Thus, the task of the model is to fit a delta impulse function.
-> How to fit?  




- Done: Figure out what the fuck is space wrapping
- Basically a mapping function to map R3 space to R3, while "density" of the space is porpotional to the sampling density of sensors

- Done: try to choose smaller area of data for reconstruction, see how it perform
- Tried on choosing structure nearby only, poor reconstruction problem still exist. Potentially due to the model don't know what to do when there is no ray (truncated due to it being very far)
- Review other's work on how to handle this problem (Mip NeRF 360)



