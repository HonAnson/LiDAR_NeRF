### Instruction to visualize data using rviz in docker

1. Build image from `noetic_base.dockerfile`:

```
docker build 
```


2. Build 3 containers from the built image, note that the containers shall be added into the same network (`host` in this case), and root should be binded to directory on your machine. I named the containers `rviz_container`, `roscore_container` and `noetic_cuda_container`.

In terminal (make sure you choose container name and directory):
``` bash
# in your terminal:
xhost +local:root && sudo docker run -it --privileged --gpus all -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --network host --add-host host.docker.internal:host-gateway --name <your container name> -v <your directory>:/root/data noetic_cuda:latest /bin/bash
```

If you have exitted the containers, you'll have to start them again. In three separate terminals, start and attach the containers
In terminal:
```bash
docker start rviz_container 
xhost +local:root
docker attach rviz_container
```

1. Copy the data into `noetic_cuda_container`


4. In `roscore` container, start roscore:
```bash
roscore
```

5. In `rviz_container`
```bash
rosrun rviz rviz
```

6. In `noetic_cuda_conatiner`
```bash
rosbag play data.bag
```



