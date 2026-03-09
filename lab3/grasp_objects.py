import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from hello_helpers.hello_misc import HelloNode
import threading
import tf2_ros
from tf2_geometry_msgs import TransformStamped, do_transform_pose_stamped
from sensor_msgs.msg import JointState
import ik_ros_utils as ik
import ikpy
import detection_utils_copy as detection_utils

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKTargetFollowing(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.03
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()
        self._grasp_done = False
    
    def joint_states_callback(self, msg):
        # unpacks joint state messages for what works with/is expected by ikpy
        with self.joint_states_lock:
            joint_states = msg
        # extract information needed for ik_solver
        joint_names = [
            'joint_lift', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'
        ]
        self.joint_state = {}
        for joint_name in joint_names:
            i = joint_states.name.index(joint_name)
            self.joint_state[joint_name] = joint_states.position[i]

    def get_goal_pose_in_base_frame(self, goal_msg):
        # TODO: ------------- start --------------
        # fill with your response
        #   transform the goal pose to the base frame
        # transform = self.tf_buffer.lookup_transform(
        #     self.target_frame,
        #     goal_msg.header.frame_id,
        #     rclpy.time.Time(),
        #     timeout=rclpy.duration.Duration(seconds=1.0),
        # )
        try:
            transform = self.tf_buffer.transform(
                goal_msg,
                self.target_frame,
                rclpy.duration.Duration(seconds=1.0)
            )
            # goal_transformed = do_transform_pose_stamped(goal_msg, transform)
            # TODO: -------------- end ---------------

            return transform
        except:
            print("Error looking up goal transform")
            return None

    def get_gripper_pose_in_base_frame(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            return transform
        except:
            print("Error looking up gripper transform")
            return None
        # xyz_out =[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z ]
        # new_msg = detection_utils.get_pose_msg(transform.header.stamp, transform.header.frame_id, xyz_out)

        # gripper_transformed = do_transform_pose_stamped(new_msg, transform)
        # gripper_transformed = new_msg
        
        # TODO: -------------- end ---------------

        # return transform

    def goal_callback(self, goal_msg):
        if self._grasp_done:
            return
        try:
            goal_transformed = self.get_goal_pose_in_base_frame(goal_msg)
            gripper_transformed = self.get_gripper_pose_in_base_frame()
            goal_pos = ik.get_xyz_from_msg(goal_transformed)
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
            print("Goal in base_link:", goal_pos)
            print("Gripper in base_link:", gripper_pos)
        except Exception as e:
            print("Error getting transforms:", e)
            return

        grasp_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/2)
        q_init = ik.get_current_configuration(self.joint_state)
        q_soln = ik.get_grasp_goal(goal_pos, grasp_orient, q_init)
        ik.print_q(q_soln)
        if q_soln is None:
            return
        ik.move_to_configuration(self, q_soln.copy())
        self.move_to_pose({'gripper_aperture': 0.0}, blocking=True)
        with self.joint_states_lock:
            lift = self.joint_state.get('joint_lift', 0.8)
            arm = self.joint_state.get('joint_arm_l0', 0.0)
        lift_up = min(1.1, lift + 0.15)
        self.move_to_pose({'joint_lift': lift_up}, blocking=True)
        arm_retract = max(0.0, arm - 0.12)
        self.move_to_pose({'joint_arm': arm_retract}, blocking=True)
        self._grasp_done = True

    def compute_waypoint_to_goal(self, goal_pos, gripper_pos):

        # TODO: ------------- start --------------
        # fill with your response
        #   find the distance between the published goal position and the gripper position
        #   if its above some threshold (delta), consider the goal to be too far (since we're trying to track the object
        #   at least 2Hz) to reach before the next goal is published
        #   in this case, find a waypoint toward the goal position that is delta away from the gripper position (make some progress towards the goal)
        #   otherwise, the goal is close and we can move there directly

        waypoint_pos = goal_pos
        dist = np.linalg.norm(goal_pos - gripper_pos)
        if dist > self.delta:
            # goal is too far
            magnitude = (goal_pos - gripper_pos) / dist
            waypoint_pos = gripper_pos + (magnitude * self.delta)
        print(waypoint_pos)
        # TODO: -------------- end ---------------

        # use an zero rotation for the waypoint (its a point so we don't need to worry about orientation)
        waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0) # [roll, pitch, yaw]

        return waypoint_pos, waypoint_orient


    def move_to_ready_pose(self):
        self.move_to_pose(ik.READY_POSE_P2, blocking=True)

    def main(self):
        HelloNode.main(self, 'follow_target', 'follow_target', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        self.joint_states_subscriber = self.create_subscription(JointState, '/stretch/joint_states', callback=self.joint_states_callback, qos_profile=1)
        self.stow_the_robot()
        self.move_to_ready_pose()
        print("At Ready Pose")


        # TODO: ------------- start --------------
        # fill with your response
        #   create a tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.goal_sub = self.create_subscription(PoseStamped, '/object_detector/goal_pose', self.goal_callback, 10)
        # TODO: -------------- end ---------------


if __name__ == '__main__':
    target_follower = IKTargetFollowing()
    target_follower.main()
    target_follower.new_thread.join()
