import cv2
import yaml
import rclpy
import os.path as osp
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import detection_utils
import message_filters
import numpy as np


# Don't forget to start the camera before starting this node!
# Part 1: using in-gripper camera
#    ros2 launch stretch_core d405_basic.launch.py
# Part 2: using head camera
#    ros2 launch stretch_core d435i_low_resolution.launch.py
#
# ros2 run rviz2 rviz2 -d `ros2 pkg prefix --share stretch_calibration`/rviz/stretch_simple_test.rviz


class YOLOEObjectDetector(Node):
    def __init__(self, obj_queries):
        super().__init__('yoloe_object_detector')
        self.visualize = True

        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.color_cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.latest_color = None
        self.latest_depth = None
        self.latest_color_cam_info = None
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.color_cam_info_sub],
            queue_size=10,
            slop=0.01
        )
        self.synchronizer.registerCallback(self.image_callback)
        self.bridge = CvBridge()
        model_path = '/home/hello-robot/models'
        model_name = 'yoloe-26s-seg.pt'
        self.model = YOLO(osp.join(model_path, model_name))
        self.obj_queries = obj_queries
        self.model.set_classes(self.obj_queries)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.publish_goals_callback)
        self.goal_pub = self.create_publisher(PoseStamped, '/object_detector/goal_pose', 10)
        self.goal_pose_msg = None

    def image_callback(self, color_msg, depth_msg, color_cam_info_msg):
        # TODO: ------------- start --------------
        try:
            self.latest_color = cv2.rotate(self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough'), cv2.ROTATE_90_CLOCKWISE)
            self.latest_depth = cv2.rotate(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'), cv2.ROTATE_90_CLOCKWISE)
            self.latest_color_cam_info = color_cam_info_msg
        except:
            print("Frames missing, skipping this callback")
        # TODO: -------------- end ---------------


    def publish_goals_callback(self):
        # TODO: ------------- start --------------
        results = self.model.predict(self.latest_color)
        detections = detection_utils.parse_results(results)
        # TODO: -------------- end ---------------
        if self.visualize:
            detection_utils.visualize_detections_masks(
                part=2, detections=detections, rgb_image=self.latest_color, depth_image=self.latest_depth)
        self.get_goal_pose(detections)
        if self.goal_pose_msg is None:
            print("OBJECT NOT DETECTED, no pose to publish")
            return
        else:
            self.goal_pub.publish(self.goal_pose_msg)
            print()
            print("---------- Published Goal Pose ----------")

    def get_goal_pose(self, detections, target_idx=0):
        if detections is None or len(detections) == 0:
            self.goal_pose_msg = None
            return None

        cam_info = self.latest_color_cam_info
        msk = detections[target_idx]["mask"]
        xyz, _ = detection_utils.mask_to_3d_centroid(self.latest_depth, cam_info, msk, fill_missing_depth=True)
        if xyz is None:
            self.goal_pose_msg = None
            return None
        timestamp = self.latest_color_cam_info.header.stamp
        frame_id = self.latest_color_cam_info.header.frame_id
        print("Goal in camera frame:", xyz)
        print("frame id:", frame_id)
        xyz[0] -= 0.1
        self.goal_pose_msg = detection_utils.get_pose_msg(timestamp, frame_id, xyz)
        return self.goal_pose_msg

if __name__ == '__main__':
    rclpy.init()
    with open('object_queries.yaml', 'r') as file:
        config = yaml.safe_load(file)
        obj_queries = config['queries']

    yolo_object_detector = YOLOEObjectDetector(obj_queries)
    rclpy.spin(yolo_object_detector)
    yolo_object_detector.destroy_node()
    rclpy.shutdown()
