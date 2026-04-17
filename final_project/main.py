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
from navigation import navigate_to_locations
from color_segmentation import ColorSegmentationDetector
from grasp import GraspNode


TASK = "Put the dishes in the sink."
MAX_GRASP_ATTEMPTS = 3
DETECT_TIMEOUT = 30.0  # seconds to wait for a valid detection


def run_grasp_node(grasp_node):
    grasp_node.main()
    grasp_node.new_thread.join()


def main():
    # --- Step 1: LLM route extraction ---
    # TODO: Call LLM and get response 
    # client = gemini_init()
    # route_response = prompt_gemini(client, "loc")
    # TODO: parse route_response.text into an ordered list of location names


    # For now, hardcode the expected route for the task above
    route = ["KITCHEN"]
    print(f"Route: {route}")

    # --- Step 2: Navigate to locations ---
    success = navigate_to_locations(route)
    if not success:
        print("Navigation failed, aborting.")
        return
    print("Navigating / navigated to the kitchen!!")

    # --- Step 3: Detect object (sequential — spin until we get a pose) ---
    rclpy.init()

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
            grasp_success = False
        else:
            # grasp_response = prompt_gemini(client, "grip")
            # TODO:  check_grasp(client, head_frame)
        
            # grasp_success = "YES" or sth similar or has Yes in the string

            grasp_success = True # TODO: just run it once for now for testing 

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
    print("TODO: implement drop-off step.")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
