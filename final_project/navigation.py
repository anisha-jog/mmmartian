#! /usr/bin/env python3

# Adapted from lab4/move_plate.py
# Navigates the robot to an ordered list of named locations using Nav2.

from copy import deepcopy
import threading
import datetime

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor

import cv2


# [x, y, orientation_z, orientation_w]
LOCATIONS = {
    'LIVING_ROOM': [-6.2514, -0.85,  0.19,  0.98],
    'KITCHEN':     [-1.06,    1.45,  0.904, 0.426],
    'HALLWAY':     [-3.33,    0.303, 0.2,   0.978],
    'SINK':        [-0.89,    1.14,  0.285, 0.959]
}

INITIAL_POSE = LOCATIONS['LIVING_ROOM']


class HeadCameraCapture(Node):
    """Captures a single frame from the head camera on demand."""
    CAMERA_TOPIC = '/camera/color/image_raw'

    def __init__(self):
        super().__init__('head_camera_capture')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.create_subscription(Image, self.CAMERA_TOPIC, self._cb, 10)

    def _cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Frame error: {e}')

    def get_frame(self):
        return self.latest_frame


def _make_pose(navigator, pt):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = pt[0]
    pose.pose.position.y = pt[1]
    pose.pose.orientation.z = pt[2]
    pose.pose.orientation.w = pt[3]
    return pose


def navigate_to_locations(location_names: list[str], timeout_s: float = 600.0):

    # rclpy.init()
    navigator = BasicNavigator()

    # Set initial pose
    init_pt = INITIAL_POSE
    initial_pose = _make_pose(navigator, init_pt)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()

    waypoints = []
    for name in location_names:
        pt = LOCATIONS.get(name.upper())
        if pt is None:
            print(f"Unknown location: {name}, skipping.")
            continue
        waypoints.append(_make_pose(navigator, pt))

    if not waypoints:
        print("No valid waypoints.")
        rclpy.shutdown()
        return False

    print(f"Navigating to: {location_names}")
    nav_start = navigator.get_clock().now()
    navigator.followWaypoints(waypoints)

    i = 0
    while not navigator.isTaskComplete():
        i += 1
        feedback = navigator.getFeedback()
        if feedback and i % 5 == 0:
            navigator.get_logger().info(
                f'Waypoint {feedback.current_waypoint + 1}/{len(waypoints)}'
            )
        if navigator.get_clock().now() - nav_start > Duration(seconds=timeout_s):
            navigator.cancelTask()
            break

    result = navigator.getResult()
    rclpy.shutdown()

    if result == TaskResult.SUCCEEDED:
        print("Navigation complete.")
        return True
    elif result == TaskResult.CANCELED:
        print("Navigation canceled.")
        return False
    else:
        print("Navigation failed.")
        return False

def get_locations():
    return LOCATIONS.keys()