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
# save a map: ros2 run nav2_map_server map_saver_cli -f martian_map


TASK = "Put the dishes in the sink."
MAX_GRASP_ATTEMPTS = 3
DETECT_TIMEOUT = 30.0  # seconds to wait for a valid detection


def run_grasp_node(grasp_node):
    grasp_node.main()
    grasp_node.new_thread.join()


def main():
    robot = stretch_body.robot.Robot()
    robot.startup()

    # # --- Step 1: LLM route extraction ---
    # # Call LLM and get response
    # client = gemini_init()
    # route_response = prompt_gemini(client, "loc", task=TASK)

    # route = None
    # if route_response is None:
    #     print("No route selected. Proceeding with default location.")
    #     route = ["KITCHEN"]
    # else:
    #     route = route_response.text
    
    #     if route not in get_locations():
    #         print("Gemini response is not in the correct format. Proceeding with default location.")
    #         route = "KITCHEN"

    # # For now, hardcode the expected route for the task above
    # # route = ["HALLWAY", "KITCHEN"]
    # # print(f"Route: {route}")

    # # --- Step 2: Navigate to locations ---
    # success = navigate_to_locations([route])
    # if not success:
    #     print("Navigation failed, aborting.")
    #     return
    # print("Navigating / navigated to the kitchen!!")

    # --- Step 3: Detect object (sequential — spin until we get a pose) ---

    print("Rotating head camera")
    # rotate head and camera
    robot.head.move_by('head_pan', np.radians(-90))
    robot.head.move_by('head_tilt', np.radians(-90))
    robot.push_command()
    # robot.wait_command()

    print("Head camera rotated")

    return
    
    rclpy.init()

    print("Starting color segmentation")
    detector = ColorSegmentationDetector()
    print("Waiting for object detection...")
    elapsed = 0.0
    while detector.latest_goal_pose is None and elapsed < DETECT_TIMEOUT:
        rclpy.spin_once(detector, timeout_sec=0.1)
        elapsed += 0.1
    goal_pose = detector.latest_goal_pose
    detector.destroy_node()

    if goal_pose is None:
        print("Object not detected within timeout, aborting.")
        rclpy.shutdown()
        return
    print(f"Object detected: {goal_pose.pose.position}")

    # --- Step 4 & 5: Grasp + verify loop ---
    grasp_node = GraspNode()
    grasp_thread = threading.Thread(target=run_grasp_node, args=(grasp_node,), daemon=True)
    grasp_thread.start()
    time.sleep(2.0)  # wait for grasp node to reach ready pose

    grasp_success = False
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
        print(f"Grasp attempt {attempt}/{MAX_GRASP_ATTEMPTS}")
        grasp_node.reset_for_retry()
        grasp_node.goal_callback(goal_pose)

        if not grasp_node._grasp_done:
            print("Grasp did not complete, retrying.")
            continue

        # --- Step 5: VLM grasp check ---
        head_frame = grasp_node.get_head_frame()
        if head_frame is None:
            print("No head camera frame available for grasp check.")
        else:
            print("Asking Gemini if the object has been gripped...")
            grip_response = prompt_gemini(client, "grip", img=head_frame)
            grasp_success = True if grip_response == "YES" else False

        # test
        grasp_success = True

        if grasp_success:
            print("Grasp verified! Proceeding to drop.")
            break
        else:
            print("Grasp failed, retrying.")
    else:
        print("Max grasp attempts reached, aborting.")
        rclpy.shutdown()
        return

    # --- Step 6: Drop object at destination (stub) ---
    # grasp_node.move_to_drop_location()
    # grasp_node.release_object()

    # move to drop location
    deposit_pt = ["SINK"]
    success_2 = navigate_to_locations(deposit_pt)
    if not success_2:
        print("Navigation failed, aborting.")
        return
    print("Navigating / navigated to the sink!!")

    # open the gripper and release object over the drop location
    robot.end_of_arm.move_to('stretch_gripper', np.radians(100))
    robot.push_command()
    robot.wait_command()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
