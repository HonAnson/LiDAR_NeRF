import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import json

# Open the bag file
file_name = r'small_plant1'
path = r'datasets/hand_collected2/' + file_name + r'.bag'
output_path = file_name + r'.json'


bag = rosbag.Bag(path)
# read point cloud and write to dict
count = 0
output = {}
for topic, msg, t in bag.read_messages():
    if topic == '/livox/lidar':
        pc = msg
        points = pc2.read_points(pc, field_names=("x", "y", "z"), skip_nans=True)
        output[count] = []
        for point in points:
            output[count].append(point)
        count += 1
bag.close()


# save dictionary into json file
with open(output_path, "w") as outfile: 
    json.dump(output, outfile)






