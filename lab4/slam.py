#! /usr/bin/env python3

# Adapted from the simple commander demo examples on
# https://github.com/ros-planning/navigation2/blob/galactic/nav2_simple_commander/nav2_simple_commander/demo_security.py

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
import scipy

"""
Basic security route patrol demo. In this demonstration, we use the D435i camera
mounted on the robot to relay the camera feed back to us that can be monitored
using RViz.
"""

# build a map: ros2 launch stretch_nav2 offline_mapping.launch.py teleop_type:=keyboard
# to load existing map: ros2 launch stretch_nav2 navigation.launch.py map:=./maps/martian_map.yaml
# save a map: ros2 run nav2_map_server map_saver_cli -f martian_map
# map names: martian_map
# pilot with xbox controller: stretch_xbox_controller_teleop.py
# pilot with keyboard: stretch_free_robot_process.py
# using head camera:
#    ros2 launch stretch_core d435i_low_resolution.launch.py
# using rviz:
# ros2 run rviz2 rviz2 -d `ros2 pkg prefix --share stretch_calibration`/rviz/stretch_simple_test.rviz

# a function to transform to quat
# scipy.spatial.transform.Rotation.from_euler('xyz', [r, p, y]).as_quat()


class CameraRecorder(Node):
    """Subscribes to the D435i head camera and writes frames to a video file."""

    CAMERA_TOPIC = '/camera/color/image_raw'
    FPS = 15.0
    FRAME_SIZE = (640, 480)  # matches d435i_low_resolution resolution

    def __init__(self, output_path):
        super().__init__('camera_recorder')
        self.bridge = CvBridge()
        self._frame_count = 0
        self._writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.FPS,
            self.FRAME_SIZE,
        )
        if not self._writer.isOpened():
            self.get_logger().error('VideoWriter failed to open — check codec/path')
        self.create_subscription(Image, self.CAMERA_TOPIC, self._image_callback, 10)
        self.get_logger().info(f'Recording head camera to {output_path}')

    def _image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(frame, self.FRAME_SIZE)
            self._writer.write(frame)
            self._frame_count += 1
            if self._frame_count % 50 == 0:
                self.get_logger().info(f'Recorded {self._frame_count} frames')
        except Exception as e:
            self.get_logger().warn(f'Frame drop: {e}')

    def stop(self):
        self._writer.release()
        self.get_logger().info(f'Camera recording saved — {self._frame_count} frames total.')


def main():
    rclpy.init()

    navigator = BasicNavigator()

    # Start head camera recorder in a background thread
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    recorder = CameraRecorder(output_path=f'patrol_{timestamp}.avi')
    recorder_executor = SingleThreadedExecutor()
    recorder_executor.add_node(recorder)
    recorder_thread = threading.Thread(target=recorder_executor.spin, daemon=True)
    recorder_thread.start()

    # Security route, probably read in from a file for a real application
    # from either a map or drive and repeat.

    # Would add here orientation coordinates after x and y
    security_route = [[-6.677, 0.056, 0.21, 0.977],
                      [-6.022, -1.43, -0.51, -1.43 ],
                      [-4.13, 0.077, 0.204, 0.978], 
                      [-2.76, 0.730, 0.277, 0.690]
    ]
    # Set our demo's initial pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = -6.677
    initial_pose.pose.position.y = 0.056
    initial_pose.pose.orientation.z = 0.21
    initial_pose.pose.orientation.w = 0.977

    navigator.setInitialPose(initial_pose)
    
    # Wait for navigation to fully activate
    navigator.waitUntilNav2Active()

    # Do security route until dead
    while rclpy.ok():
        # Send our route
        route_poses = []
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = navigator.get_clock().now().to_msg()

        for pt in security_route[1:]:
            # set orientation here when we have it
            rotation = [0, 0, pt[2]]
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = pt[2]
            pose.pose.orientation.w = pt[3]

            route_poses.append(deepcopy(pose))
        
        nav_start = navigator.get_clock().now()
        navigator.followWaypoints(route_poses)

        # Do something during our route (e.x. AI detection on camera images for anomalies)
        # Simply print ETA for the demonstation
        i = 0
        while not navigator.isTaskComplete():
            i += 1
            feedback = navigator.getFeedback()
            if feedback and i % 5 == 0:
                navigator.get_logger().info('Executing current waypoint: ' +
                    str(feedback.current_waypoint + 1) + '/' + str(len(route_poses)))
                now = navigator.get_clock().now()

                # Some navigation timeout to demo cancellation
                if now - nav_start > Duration(seconds=600.0):
                    navigator.cancelTask()

        # If at end of route, reverse the route to restart
        security_route.reverse()

        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            navigator.get_logger().info('Route complete! Restarting...')
        elif result == TaskResult.CANCELED:
            navigator.get_logger().info('Security route was canceled, exiting.')
            recorder.stop()
            rclpy.shutdown()
        elif result == TaskResult.FAILED:
            navigator.get_logger().info('Security route failed! Restarting from other side...')

    recorder.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()