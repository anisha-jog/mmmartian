#! /usr/bin/env python3

# Color-segmentation-based object detector.
# Replaces YOLO from lab3/object_detector.py with HSV color thresholding.
# Publishes a goal PoseStamped on /object_detector/goal_pose, same interface
# as YOLOEObjectDetector so grasp.py can subscribe without changes.

import sys
sys.path.insert(0, '/home/tiffany/mmmartian/lab3')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import detection_utils  # from lab3


# HSV range for the target object — tune these for your object's color.
# Example: a blue dish (H: 100-130, S: 80-255, V: 50-255)
HSV_LOWER = np.array([100, 80, 50])
HSV_UPPER = np.array([130, 255, 255])

MIN_CONTOUR_AREA = 500  # pixels, ignore noise below this


def segment_by_color(bgr_image: np.ndarray):
    """Return (centroid_xy, mask_polygon) or (None, None) if no object found.

    centroid_xy: (x, y) pixel tuple
    mask_polygon: Nx2 int array of contour points
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None, None

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None, None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    polygon = largest.reshape(-1, 2)
    return (cx, cy), polygon


class ColorSegmentationDetector(Node):
    """Drop-in replacement for YOLOEObjectDetector using color segmentation."""

    def __init__(self):
        super().__init__('color_segmentation_detector')

        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/color/camera_info')

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.cam_info_sub],
            queue_size=10, slop=0.05
        )
        self.synchronizer.registerCallback(self._image_callback)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_cam_info = None

        self.goal_pub = self.create_publisher(PoseStamped, '/object_detector/goal_pose', 10)
        self.latest_goal_pose = None  # set each time a valid pose is computed
        self.create_timer(0.5, self._publish_goal_callback)

    def _image_callback(self, color_msg, depth_msg, cam_info_msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.latest_cam_info = cam_info_msg
        except Exception as e:
            self.get_logger().warn(f'Frame error: {e}')

    def _publish_goal_callback(self):
        if self.latest_color is None or self.latest_depth is None:
            return

        centroid_xy, mask_polygon = segment_by_color(self.latest_color)

        if centroid_xy is None:
            self.get_logger().info('No object detected by color segmentation.')
            return

        x, y = centroid_xy
        z_depth = float(self.latest_depth[y, x])
        if z_depth == 0:
            # try mask centroid from 3D projection instead
            xyz, _ = detection_utils.mask_to_3d_centroid(
                self.latest_depth, self.latest_cam_info, mask_polygon)
            if xyz is None:
                return
        else:
            xyz = detection_utils.pixel_to_3d(centroid_xy, z_depth, self.latest_cam_info)

        timestamp = self.latest_cam_info.header.stamp
        frame_id = self.latest_cam_info.header.frame_id
        goal_msg = detection_utils.get_pose_msg(timestamp, frame_id, xyz)
        self.latest_goal_pose = goal_msg
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f'Published goal: {xyz}')


if __name__ == '__main__':
    rclpy.init()
    node = ColorSegmentationDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
