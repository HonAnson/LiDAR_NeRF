import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import json



if __name__ == '__main__':
    # Open the bag file
    file_name = r'small_plant1'
    path = r'datasets/hand_collected2/' + file_name + r'.bag'


    bag = rosbag.Bag(path)
    # read point cloud and write to dict
    count = 0
    start_frame = 0
    output = {}
    for topic, msg, t in bag.read_messages():
        if topic == '/livox/lidar':
            # ### for reading livox bag files
            # pc = msg.points
            # output[count] = []
            # for point in pc:
            #     point_pos = [point.x, point.y, point.z]
            #     output[count].append(point_pos)
            # count += 1
            
            ## for reading "normal" bag files
            pc = msg
            points = pc2.read_points(pc, field_names=("x", "y", "z"), skip_nans=True)
            output[count] = []
            for point in points:
                output[count].append(point)
            count += 1
            
            # save dictionary into json file every 10 frame
            if count % 100 == 0:
                output_path = r'datasets/json/' + file_name + r'/' + file_name + r'_frame' + str(start_frame) + r'_' + str(count) + r'.json'
                with open(output_path, "w") as outfile: 
                    json.dump(output, outfile)
                output = {}
                start_frame = count

    output_path = r'datasets/json/' + file_name + r'/' + file_name + r'_frame' + str(start_frame) + r'_' + str(count) + r'.json'
    with open(output_path, "w") as outfile: 
        json.dump(output, outfile)
    
    
    bag.close()







