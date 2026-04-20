#! /usr/bin/env python3

# Top-level pipeline for the final project.
#
# Pipeline:
#   1. Parse natural-language task → ordered location route  (instruction_query)
#   2. Navigate to each location in route                    (navigation)
#   3. Detect target object by color segmentation            (color_segmentation)
#   4. Attempt IK grasp                                      (grasp)
#   5. VLM grasp verification via head camera                (instruction_query)
#      → retry grasp up to MAX_GRASP_ATTEMPTS if failed
#   6. [TODO] Drop object at destination                     (grasp stubs)

import threading
import time

import cv2
import rclpy

from instruction_query import gemini_init, prompt_gemini
from navigation import navigate_to_locations, get_locations
from color_segmentation import ColorSegmentationDetector
from grasp import GraspNode

import stretch_body.robot
import numpy as np

# launch ROS:
#    ros2 launch stretch_core stretch_driver.launch.py
# load existing map (launches ROS automatically):
#    ros2 launch stretch_nav2 navigation.launch.py map:=./maps/martian_ai_space.yaml
# using head camera:
#    ros2 launch stretch_core d435i_low_resolution.launch.py
# using in-gripper camera:
#    ros2 launch stretch_core d405_basic.launch.py
# build a map: ros2 launch stretch_nav2 offline_mapping.launch.py teleop_type:=keyboard
# save a map: ros2 run nav2_map_server map_saver_cli -f martian_map_new


TASK = "Put the dishes in the sink."
MAX_GRASP_ATTEMPTS = 3
DETECT_TIMEOUT = 30.0  # seconds to wait for a valid detection


def run_grasp_node(grasp_node):
    grasp_node.main()
    grasp_node.new_thread.join()


def stream_head_camera(grasp_node, stop_event):
    """Display head camera frames with segmentation centroid overlay until stop_event is set."""
    cv2.namedWindow('Head Camera', cv2.WINDOW_NORMAL)
    while not stop_event.is_set():
        frame = grasp_node.get_head_frame()
        if frame is not None:
            display = frame.copy()
            centroid = grasp_node.latest_centroid
            if centroid is not None:
                cx, cy = centroid
                cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                cv2.drawMarker(display, (cx, cy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
            cv2.imshow('Head Camera', display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()


def main():
    # robot = stretch_body.robot.Robot()
    # robot.startup()

    gemini_mode = False

    # --- Step 1: LLM route extraction ---
    client = None
    route = ["KITCHEN"]
    print("starting step one - LLM route extraction")
    if gemini_mode:
        # Call LLM and get response
        client = gemini_init()
        print("Gemini client initialized, prompting for route...")
        route_response = prompt_gemini(client, "loc", task=TASK)
        print(f"Gemini response: {route_response}")
        if route_response is None:
            print("No route selected. Proceeding with default location.")
        else:
            if route_response.text not in get_locations():
                print("Gemini response is not in the correct format. Proceeding with default location.")
            else:
                route = [route_response.text]
    else:
        print("Gemini mode is disabled. Proceeding with default location.")

    # # --- Step 2: Navigate to locations ---
    # rclpy.init()
    # success = navigate_to_locations(route)
    # if not success:
    #     print("Navigation failed, aborting.")
    #     return
    # print("Navigated to the task location!!")

    # # rclpy.shutdown()  # close first context so GraspNode (inherits HelloNode) can call rclpy.init() cleanly
    # --- Step 3: Detect object (sequential — spin until we get a pose) ---
    grasp_node = GraspNode()
    grasp_thread = threading.Thread(target=run_grasp_node, args=(grasp_node,), daemon=True)
    grasp_thread.start()
    grasp_node._initialized.wait()  # block until subscriptions and TF are fully set up
    grasp_node.switch_to_position_mode()
    time.sleep(1.0)  # let joint state callbacks populate before the first grasp reads them

    # rotate head and camera
    print("Rotating head camera")
    grasp_node.rotate_camera(-90, -45)
    time.sleep(2.0) # let the camera actually rotate before detection starts
    print("Head camera rotated")

    # test joints
    grasp_node.start_position()

    return
    
    # rclpy.init()

    print("Starting color segmentation")
    detector = ColorSegmentationDetector()
    print("Waiting for object detection...")

    cv2.namedWindow('Head Camera (detecting)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Mask', cv2.WINDOW_NORMAL)

    elapsed = 0.0
    while detector.latest_goal_pose is None and elapsed < DETECT_TIMEOUT:
        rclpy.spin_once(detector, timeout_sec=0.1)
        elapsed += 0.1

        if detector.latest_color is not None:
            display = detector.latest_color.copy()
            hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
            from color_segmentation import HSV_LOWER, HSV_UPPER, segment_by_color
            mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
            centroid_xy, _ = segment_by_color(display)
            if centroid_xy is not None:
                cx, cy = centroid_xy
                cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow('Head Camera (detecting)', display)
            cv2.imshow('HSV Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    goal_pose = detector.latest_goal_pose
    detector.destroy_node()
    # rclpy.shutdown()  # close first context so HelloNode.main() can call rclpy.init() cleanly

    if goal_pose is None:
        print("Object not detected within timeout, aborting.")
        return
    print(f"Object detected: {goal_pose.pose.position}")

    # --- Step 4 & 5: Grasp + verify loop ---
    # grasp_node = GraspNode()
    # grasp_thread = threading.Thread(target=run_grasp_node, args=(grasp_node,), daemon=True)
    # grasp_thread.start()
    # grasp_node._initialized.wait()  # block until subscriptions and TF are fully set up
    # grasp_node.switch_to_position_mode()
    # time.sleep(1.0)  # let joint state callbacks populate before the first grasp reads them

    camera_stop = threading.Event()
    camera_thread = threading.Thread(target=stream_head_camera, args=(grasp_node, camera_stop), daemon=True)
    camera_thread.start()

    grasp_success = False
    grasp_node.reset_for_retry()
    
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
        print(f"Grasp attempt {attempt}/{MAX_GRASP_ATTEMPTS}")
        grasp_node.reset_for_retry()
        print("grasp reset")
        grasp_node.goal_callback(goal_pose)
        print("grasp attempted")
        print(grasp_node._grasp_done)

        if not grasp_node._grasp_done:
            print("Grasp did not complete, retrying.")
            continue

        # --- Step 5: VLM grasp check ---
        head_frame = grasp_node.get_head_frame()
        if head_frame is None:
            print("No head camera frame available for grasp check.")
        elif gemini_mode:
            print("Asking Gemini if the object has been gripped...")
            grip_response = prompt_gemini(client, "grip", img=head_frame)
            grasp_success = True if grip_response == "YES" or grip_response is None else False
            print("Grip is successful OR Gemini cannot parse the grasp. Proceeding with navigation.")
        else:
            print("Gemini mode is not enabled. No feedback will be requested.")

        # test
        # grasp_success = True

        if grasp_success:
            print("Grasp verified! Proceeding to drop.")
            break
        else:
            print("Grasp failed, retrying.")
    else:
        print("Max grasp attempts reached, aborting.")
        rclpy.shutdown()
        return

    # move to drop location
    grasp_node.switch_to_navigation_mode()

    # hardcoded deposit point
    deposit_pt = ["SINK"]
    success_2 = navigate_to_locations(deposit_pt)
    if not success_2:
        print("Navigation failed, aborting.")
        return
    print("Navigated to the deposit point!!")

    # open the gripper and release object over the drop location
    print("Extending arm and dropping object")
    grasp_node.extend_and_drop() # if hanging, Ctrl+C the ROS terminal to release the gripper

    # rclpy.shutdown()


if __name__ == '__main__':
    main()
