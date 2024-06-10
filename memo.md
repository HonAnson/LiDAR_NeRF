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
 
### Trial 2





