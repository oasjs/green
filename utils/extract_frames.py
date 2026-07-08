import os
import cv2
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# need to implement frame_skip and set it based on the wanted value from the user, given the existence of bags that do not alter in significant ways continuously.
def extract_frames_from_bag(current_dir, frame_skip=1):
    # path to .mcap file
    bag_path = current_dir
    # path to output dir
    output_dir = f"../frames/"
    # path to topic collected -> in the case of collecting frames, the raw image of the left camera
    topic_name = "/oak/left/image_raw"

    # if output_dir does not exist, creates it
    os.makedirs(output_dir, exist_ok=True)
    # defines bridge as OpenCV bridge
    bridge = CvBridge()

    # rosbag2 reader for storing mcap bags
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="mcap"
    )
    
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # obtains message type
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    msg_type = type_map.get(topic_name)

    if msg_type is None:
        print(f"Topic '{topic_name}' was not found.")
        return

    msg_class = get_message(msg_type)

    i = 0
    print(f"Extraindo frames de '{bag_path}' do tópico '{topic_name}'...")

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic == topic_name:
            msg = deserialize_message(data, msg_class)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = os.path.join(output_dir, f"frame_{i:06d}.png")
            cv2.imwrite(filename, cv_img)
            i += 1

    print(f"Extracted {i} frames to '{output_dir}'")


if __name__ == "__main__":
    directories = os.listdir("../bags")
    for directory in directories:
        extract_frames_from_bag(os.path.join("../bags", directory))
