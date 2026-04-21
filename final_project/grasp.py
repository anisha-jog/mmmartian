#! /usr/bin/env python3

# Adapted from lab3/grasp_objects.py
# IK-based grasping node. Subscribes to /object_detector/goal_pose
# (published by color_segmentation.py) and executes a grasp.
# After grasp, signals main.py via a shared flag so grasp verification can run.

import sys
sys.path.insert(0, '/homffany/mmmartian/lab3')

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from geometry_msgs.msg import PoseStamped
from hello_helpers.hello_misc import HelloNode, Node
import threading
import tf2_ros
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import ik_ros_utils as ik
import ikpy
from color_segmentation import segment_by_color
import time


class GraspNode(HelloNode, Node):
    """Listens for a goal pose and attempts a grasp using IK."""

    def __init__(self):
        HelloNode.__init__(self)  # HelloNode.main() calls rclpy.init() + Node.__init__ itself
        self.delta = 0.03
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()
        self.joint_state = {}
        self._grasp_done = False
        self._initialized = threading.Event()

        # Latest head camera frame for grasp verification
        self.bridge = CvBridge()
        self.latest_head_frame = None
        self.latest_centroid = None  # (x, y) pixel from color segmentation

    # ------------------------------------------------------------------ #
    #  Joint state tracking (unchanged from lab3)                          #
    # ------------------------------------------------------------------ #

    def joint_states_callback(self, msg):
        with self.joint_states_lock:
            joint_names = [
                'joint_lift', 'joint_arm_l0',
                'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll',
                'joint_head_pan', 'joint_head_tilt'
            ]
            self.joint_state = {}
            for name in joint_names:
                if name in msg.name:
                    i = msg.name.index(name)
                    self.joint_state[name] = msg.position[i]

    # ------------------------------------------------------------------ #
    #  TF helpers (unchanged from lab3)                                    #
    # ------------------------------------------------------------------ #

    def get_goal_pose_in_base_frame(self, goal_msg):
        try:
            # Stamp with time=0 so TF2 uses the latest available transform
            # rather than extrapolating back to the original detection timestamp.
            stamped = PoseStamped()
            stamped.header.frame_id = goal_msg.header.frame_id
            stamped.header.stamp = rclpy.time.Time().to_msg()
            stamped.pose = goal_msg.pose
            return self.tf_buffer.transform(
                stamped, self.target_frame,
                rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            print("Error transforming goal pose:", e)
            return None

    def get_gripper_pose_in_base_frame(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame, self.gripper_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except Exception:
            print("Error looking up gripper pose")
            return None

    # ------------------------------------------------------------------ #
    #  Head camera capture                                                 #
    # ------------------------------------------------------------------ #

    def _head_camera_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_head_frame = frame
            centroid, _ = segment_by_color(frame)
            self.latest_centroid = centroid  # (x, y) or None
        except Exception as e:
            self.get_logger().warn(f'Head camera frame error: {e}')

    def get_head_frame(self):
        """Return the most recent BGR frame from the head camera."""
        return self.latest_head_frame

    # ------------------------------------------------------------------ #
    #  Grasp execution (adapted from lab3)                                 #
    # ------------------------------------------------------------------ #


    def goal_callback(self, goal_msg):
        if self._grasp_done:
            return

        goal_transformed = self.get_goal_pose_in_base_frame(goal_msg)
        gripper_transformed = self.get_gripper_pose_in_base_frame()
        if goal_transformed is None or gripper_transformed is None:
            return

        goal_pos = ik.get_xyz_from_msg(goal_transformed)
        goal_pos[1] -= 0.10  # approach offset
        goal_pos[2] = min(goal_pos[2], 1.0) 
        gripper_pos = ik.get_xyz_from_msg(gripper_transformed)

        grasp_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi / 2)
        q_init = ik.get_current_configuration(self.joint_state)
        print("init complete")
        q_soln = ik.get_grasp_goal(goal_pos, grasp_orient, q_init)
        print("solution calculation complete")
        if q_soln is None:
            return

        print("Moving to grasp pose")
        ik.move_to_configuration(self, q_soln.copy())

        print("move to config called", q_soln.copy())

        print("Closing gripper")
        self.move_to_pose({'gripper_aperture': -0.2}, blocking=True)
        print("Gripper closed")

        with self.joint_states_lock:
            lift = self.joint_state.get('joint_lift', 0.8)
            arm = self.joint_state.get('joint_arm_l0', 0.0)
        print(f"Lifting: {lift:.3f} -> {min(1.1, lift + 0.15):.3f}")
        self.move_to_pose({'joint_lift': min(1.1, lift + 0.15)}, blocking=True)
        print(f"Retracting arm: {arm:.3f} -> {max(0.0, arm - 0.12):.3f}")
        self.move_to_pose({'joint_arm': max(0.0, arm - 0.12)}, blocking=True)
        print("Grasp sequence complete")

        self._grasp_done = True

    def reset_for_retry(self):
        """Reset state so a new grasp attempt can be made."""
        self._grasp_done = False
        print("moving grasp to retry")
        self.move_to_pose(ik.READY_POSE_P2, blocking=True, timeout=3.0)

    # ------------------------------------------------------------------ #
    #  Downstream tasks (stubs)                                            #
    # ------------------------------------------------------------------ #

    def extend_and_drop(self):
        """TODO: Navigate arm/base to sink drop position."""
        with self.joint_states_lock:
            arm = self.joint_state.get('joint_arm_l0', 0.0)
        print(f"Extending arm: {arm:.3f} -> {min(1.0, arm + 0.5):.3f}")
        self.move_to_pose({'joint_arm': min(1.0, arm + 0.5)}, blocking=True, timeout=3.0) # this may hang
        print("Opening gripper")
        self.move_to_pose({'gripper_aperture': 0.5}, blocking=True)

    def rotate_camera(self, pan_deg, tilt_deg):
        """TODO: Navigate arm/base to sink drop position."""
        with self.joint_states_lock:
            head_pan = self.joint_state.get('joint_head_pan', 0.0)
            head_tilt = self.joint_state.get('joint_head_tilt', 0.0)
        print(f"Rotating camera: {head_pan:.3f} -> {np.radians(pan_deg):.3f}")
        self.move_to_pose({'joint_head_pan': np.radians(pan_deg)}, blocking=True) # this may hang
        print(f"Rotating camera: {head_tilt:.3f} -> {np.radians(tilt_deg):.3f}")
        self.move_to_pose({'joint_head_tilt': np.radians(tilt_deg)}, blocking=True)

    def start_position(self):
        self.stow_the_robot()

    # ------------------------------------------------------------------ #
    #  Node lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def main(self):
        HelloNode.main(self, 'grasp_node', 'grasp_node', wait_for_first_pointcloud=False)
        self.callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            JointState, '/stretch/joint_states', self.joint_states_callback, 1)
        self.create_subscription(
            Image, '/camera/color/image_raw', self._head_camera_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(
            PoseStamped, '/object_detector/goal_pose', self.goal_callback, 10)

        # Signal only after subscriptions and TF are fully set up
        self._initialized.set()


if __name__ == '__main__':
    node = GraspNode()
    node.main()
    node.new_thread.join()