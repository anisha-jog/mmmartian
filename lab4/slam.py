#! /usr/bin/env python3

# Adapted from the simple commander demo examples on 
# https://github.com/ros-planning/navigation2/blob/galactic/nav2_simple_commander/nav2_simple_commander/demo_security.py

from copy import deepcopy

from geometry_msgs.msg import PoseStamped
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

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

def main():
    rclpy.init()

    navigator = BasicNavigator()

    # Security route, probably read in from a file for a real application
    # from either a map or drive and repeat.

    # Would add here orientation coordinates after x and y
    security_route = [[-2.23, -1.34, 0.502, 0.864 ],
                      [0.0, -1.624, -0.279, 0.96 ], 
    ]
    # Set our demo's initial pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = -2.23
    initial_pose.pose.position.y = -1.34
    initial_pose.pose.orientation.z = 0.502
    initial_pose.pose.orientation.w = 0.864

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
            rclpy.shutdown()
        elif result == TaskResult.FAILED:
            navigator.get_logger().info('Security route failed! Restarting from other side...')

    rclpy.shutdown()


if __name__ == '__main__':
    main()