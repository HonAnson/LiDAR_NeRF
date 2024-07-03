# import rosbag
# from std_msgs.msg import String

# path = r'datasets/hand_collected2/testing.bag'
# bag = rosbag.Bag(path)
# for topic, msg, t in bag.read_messages(topics = ['/livox/lidar']):
#     print(f"Topic: {topic}, Message: {msg}, Time: {t}")

# # print(bag)
# bag.close()


import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

# Open the bag file
path = r'datasets/hand_collected2/testing.bag'
bag = rosbag.Bag(path)
# Read all messages from the bag file
count = 0
for topic, msg, t in bag.read_messages():
    if topic == '/livox/lidar':
        pc = msg
        points = pc2.read_points(pc, field_names=("x", "y", "z"), skip_nans=True)
        for point in points:
            print(f"Point: {point}")
            count += 1
# Close the bag file
print(count)
bag.close()







