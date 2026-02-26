import ikpy.urdf.utils
import urchin as urdfpy
import numpy as np
import ikpy.chain
import stretch_body.robot
import importlib.resources as importlib_resources
import hello_helpers.hello_misc as hm
import time
from scipy.spatial.transform import Rotation
# NOTE before running: `python3 -m pip install --upgrade ikpy graphviz urchin networkx scipy`

target_orientation = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/2) # [roll, pitch, yaw]

# # Setup the Python API
# robot = stretch_body.robot.Robot()
# robot.startup()
# # Ensure robot is homed
# if not robot.is_calibrated():
#     robot.home()

robot = hm.HelloNode.quick_create('robot', wait_for_first_pointcloud=False)
# 可选：启动时先移动底座并 stow；若 arm action server 未就绪会超时，可注释掉或设 SKIP_INITIAL_MOVE=1 跳过
import os
if os.environ.get('SKIP_INITIAL_MOVE', '0') != '1':
    try:
        robot.move_to_pose({'translate_mobile_base': 0.35}, blocking=True)
        robot.stow_the_robot()
    except Exception as e:
        print("[WARN] 初始移动/ stow 未执行（arm action server 可能未就绪）:", e)
        print("        可先启动仿真/机器人后再运行，或设环境变量 SKIP_INITIAL_MOVE=1 跳过。")
else:
    print("[INFO] SKIP_INITIAL_MOVE=1，跳过初始移动与 stow。")


pkg_path = str(importlib_resources.files('stretch_urdf'))
urdf_file_path = pkg_path + '/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf'

# Remove unnecessary links/joints
original_urdf = urdfpy.URDF.load(urdf_file_path)
modified_urdf = original_urdf.copy()

names_of_links_to_remove = ['link_right_wheel', 'link_left_wheel', 'caster_link', 'link_head', 'link_head_pan', 'link_head_tilt', 'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder', 'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame', 'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame', 'gripper_camera_bottom_screw_frame', 'gripper_camera_link', 'gripper_camera_depth_frame', 'gripper_camera_depth_optical_frame', 'gripper_camera_infra1_frame', 'gripper_camera_infra1_optical_frame', 'gripper_camera_infra2_frame', 'gripper_camera_infra2_optical_frame', 'gripper_camera_color_frame', 'gripper_camera_color_optical_frame', 'laser', 'base_imu', 'respeaker_base', 'link_wrist_quick_connect', 'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_aruco_fingertip_right', 'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_aruco_fingertip_left', 'link_aruco_d405', 'link_head_nav_cam']
# links_kept = ['base_link', 'link_mast', 'link_lift', 'link_arm_l4', 'link_arm_l3', 'link_arm_l2', 'link_arm_l1', 'link_arm_l0', 'link_wrist_yaw', 'link_wrist_yaw_bottom', 'link_wrist_pitch', 'link_wrist_roll', 'link_gripper_s3_body', 'link_grasp_center']
links_to_remove = [l for l in modified_urdf._links if l.name in names_of_links_to_remove]
for lr in links_to_remove:
    modified_urdf._links.remove(lr)
names_of_joints_to_remove = ['joint_right_wheel', 'joint_left_wheel', 'caster_joint', 'joint_head', 'joint_head_pan', 'joint_head_tilt', 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder', 'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'camera_joint', 'camera_link_joint', 'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint', 'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint', 'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint', 'gripper_camera_joint', 'gripper_camera_link_joint', 'gripper_camera_depth_joint', 'gripper_camera_depth_optical_joint', 'gripper_camera_infra1_joint', 'gripper_camera_infra1_optical_joint', 'gripper_camera_infra2_joint', 'gripper_camera_infra2_optical_joint', 'gripper_camera_color_joint', 'gripper_camera_color_optical_joint', 'joint_laser', 'joint_base_imu', 'joint_respeaker', 'joint_wrist_quick_connect', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right', 'joint_aruco_fingertip_right', 'joint_gripper_finger_left', 'joint_gripper_fingertip_left', 'joint_aruco_fingertip_left', 'joint_aruco_d405', 'joint_head_nav_cam'] 
# joints_kept = ['joint_mast', 'joint_lift', 'joint_arm_l4', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_yaw_bottom', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_gripper_s3_body', 'joint_grasp_center']
joints_to_remove = [l for l in modified_urdf._joints if l.name in names_of_joints_to_remove]
for jr in joints_to_remove:
    modified_urdf._joints.remove(jr)

# Add virtual base joint
joint_base_rotation = urdfpy.Joint(name='joint_base_rotation',
                                      parent='base_link',
                                      child='link_base_rotation',
                                      joint_type='revolute',
                                      axis=np.array([0.0, 0.0, 1.0]),
                                      origin=np.eye(4, dtype=np.float64),
                                      limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
modified_urdf._joints.append(joint_base_rotation)
link_base_rotation = urdfpy.Link(name='link_base_rotation',
                                    inertial=None,
                                    visuals=None,
                                    collisions=None)
modified_urdf._links.append(link_base_rotation)
joint_base_translation = urdfpy.Joint(name='joint_base_translation',
                                      parent='joint_base_rotation',
                                      child='link_base_translation',
                                      joint_type='prismatic',
                                      axis=np.array([1.0, 0.0, 0.0]),
                                      origin=np.eye(4, dtype=np.float64),
                                      limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
modified_urdf._joints.append(joint_base_translation)
link_base_translation = urdfpy.Link(name='link_base_translation',
                                    inertial=None,
                                    visuals=None,
                                    collisions=None)
modified_urdf._links.append(link_base_translation)

# amend the chain
for j in modified_urdf._joints:
    if j.name == 'joint_mast':
        j.parent = 'link_base_translation'
    # if j.name == 'joint_lift':
    #     j.parent = 'link_base_rotation'
    # maybe add here? used to be 

new_urdf_path = "/tmp/iktutorial/stretch.urdf"
modified_urdf.save(new_urdf_path)

chain = ikpy.chain.Chain.from_urdf_file(new_urdf_path)

for link in chain.links:
    print(f"* Link Name: {link.name}, Type: {link.joint_type}")

def get_current_configuration():
    def bound_range(name, value):
        # names = [l.name for l in chain.links]
        index = robot.joint_state.name.index(name)
        # print("links in chain links", chain.links)
        # print("tried to find index ", index, " for name ", name, " in robot joint state with names ", robot.joint_state.name)
        bounds = chain.links[index].bounds
        
        # print("tried to find index ", index, " for name ", name, " in robot joint state with names ", robot.joint_state.name)
        return min(max(value, bounds[0]), bounds[1])

    q_base = 0.0
    q_lift = bound_range('joint_lift', robot.joint_state.position[robot.joint_state.name.index('joint_lift')])
    # print(robot.joint_state.name, f"index of arm l0{robot.joint_state.name.index('joint_arm_l0')}")
    q_arml = bound_range('joint_arm_l0', robot.joint_state.position[robot.joint_state.name.index('joint_arm_l0')] / 4.0)
    q_yaw = bound_range('joint_wrist_yaw', robot.joint_state.position[robot.joint_state.name.index('joint_wrist_yaw')])
    q_pitch = bound_range('joint_wrist_pitch', robot.joint_state.position[robot.joint_state.name.index('joint_wrist_pitch')])
    q_roll = bound_range('joint_wrist_roll', robot.joint_state.position[robot.joint_state.name.index('joint_wrist_roll')])
    return [0.0, q_base, 0.0, q_lift, 0.0, q_arml, q_arml, q_arml, q_arml, q_yaw, 0.0, q_pitch, q_roll, 0.0, 0.0]

def move_to_configuration(q):
    q_base = q[1]
    q_base_rotation = q[2] # was it 2 or 0?
    q_lift = q[3]
    q_arm = q[5] + q[6] + q[7] + q[8]
    q_yaw = q[9]
    q_pitch = q[11]
    q_roll = q[12]
    # robot.base.translate_by(q_base)
    # robot.lift.move_to(q_lift)
    # robot.arm.move_to(q_arm)
    # robot.end_of_arm.move_to('wrist_yaw', q_yaw)
    # robot.end_of_arm.move_to('wrist_pitch', q_pitch)
    # robot.end_of_arm.move_to('wrist_roll', q_roll)
    # robot.push_command()
    # print(robot.joint_state.name)
    robot.move_to_pose({'rotate_mobile_base': q_base_rotation}, blocking=True)
    robot.move_to_pose({'translate_mobile_base': q_base}, blocking=True)
    robot.move_to_pose({'joint_lift': q_lift}, blocking=True)
    robot.move_to_pose({'joint_arm': q_arm}, blocking=True)
    robot.move_to_pose({'joint_wrist_yaw': q_yaw}, blocking=True)
    robot.move_to_pose({'joint_wrist_pitch': q_pitch}, blocking=True)
    robot.move_to_pose({'joint_wrist_roll': q_roll}, blocking=True)

def move_to_grasp_goal(target_point, target_orientation):
    q_init = get_current_configuration()
    # print('Initial configuration:', q_init)
    # print('Target point:', target_point)
    # print('Target orientation:\n', target_orientation)
    # print("len(q_init):", len(q_init))
    # print("len(active_links_mask):", len(chain.active_links_mask))
    # print("active count:", sum(chain.active_links_mask))
    q_soln = chain.inverse_kinematics(target_point, target_orientation, orientation_mode='all', initial_position=q_init)
    # print('Solution:', q_soln)

    err = np.linalg.norm(chain.forward_kinematics(q_soln)[:3, 3] - target_point)
    if not np.isclose(err, 0.0, atol=1e-2):
        print("IKPy did not find a valid solution")
        return
    move_to_configuration(q=q_soln)
    return q_soln

def get_current_grasp_pose():
    q = get_current_configuration()
    return chain.forward_kinematics(q)


# ---------- 先给出 prior p（当前抓取位姿） ----------
print("Prior p (current grasp pose 4x4 matrix):")
prior_pose = get_current_grasp_pose()
print(prior_pose)
print()

# ---------- 循环：每次输入 7 位位姿 (x y z qx qy qz qw) 并执行 ----------
print("输入 7 位位姿: x y z qx qy qz qw（单位：米 + 四元数），空行退出。")
while True:
    s = input("位姿 > ").strip()
    if s == "":
        print("退出。")
        break
    try:
        vals = [float(x) for x in s.split()]
        if len(vals) != 7:
            print("需要恰好 7 个数字 (x y z qx qy qz qw)，请重试。")
            continue
        x, y, z, qx, qy, qz, qw = vals
        target_point = [x, y, z]
        target_orientation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        move_to_grasp_goal(target_point, target_orientation)
        print("执行完成。当前位姿:\n", get_current_grasp_pose())
    except ValueError as e:
        print("解析失败，请输入 7 个数字，用空格分隔。", e)
    except Exception as e:
        print("执行出错:", e)
print("Done!")
