import urchin as urdfpy
import numpy as np
import ikpy.chain
import stretch_body.robot
import importlib.resources as importlib_resources
from scipy.spatial.transform import Rotation
# NOTE: `python3 -m pip install --upgrade ikpy graphviz urchin networkx scipy`

# ---------- Stretch Python API（不用 ROS）----------
robot = stretch_body.robot.Robot()
robot.startup()
if not robot.is_calibrated():
    robot.home()

pkg_path = str(importlib_resources.files('stretch_urdf'))
urdf_file_path = pkg_path + '/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf'

# Remove unnecessary links/joints
original_urdf = urdfpy.URDF.load(urdf_file_path)
modified_urdf = original_urdf.copy()

names_of_links_to_remove = ['link_right_wheel', 'link_left_wheel', 'caster_link', 'link_head', 'link_head_pan', 'link_head_tilt', 'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder', 'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame', 'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame', 'gripper_camera_bottom_screw_frame', 'gripper_camera_link', 'gripper_camera_depth_frame', 'gripper_camera_depth_optical_frame', 'gripper_camera_infra1_frame', 'gripper_camera_infra1_optical_frame', 'gripper_camera_infra2_frame', 'gripper_camera_infra2_optical_frame', 'gripper_camera_color_frame', 'gripper_camera_color_optical_frame', 'laser', 'base_imu', 'respeaker_base', 'link_wrist_quick_connect', 'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_aruco_fingertip_right', 'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_aruco_fingertip_left', 'link_aruco_d405', 'link_head_nav_cam']
links_to_remove = [l for l in modified_urdf._links if l.name in names_of_links_to_remove]
for lr in links_to_remove:
    modified_urdf._links.remove(lr)

names_of_joints_to_remove = ['joint_right_wheel', 'joint_left_wheel', 'caster_joint', 'joint_head', 'joint_head_pan', 'joint_head_tilt', 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder', 'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'camera_joint', 'camera_link_joint', 'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint', 'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint', 'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint', 'gripper_camera_joint', 'gripper_camera_link_joint', 'gripper_camera_depth_joint', 'gripper_camera_depth_optical_joint', 'gripper_camera_infra1_joint', 'gripper_camera_infra1_optical_joint', 'gripper_camera_infra2_joint', 'gripper_camera_infra2_optical_joint', 'gripper_camera_color_joint', 'gripper_camera_color_optical_joint', 'joint_laser', 'joint_base_imu', 'joint_respeaker', 'joint_wrist_quick_connect', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right', 'joint_aruco_fingertip_right', 'joint_gripper_finger_left', 'joint_gripper_fingertip_left', 'joint_aruco_fingertip_left', 'joint_aruco_d405', 'joint_head_nav_cam']
joints_to_remove = [j for j in modified_urdf._joints if j.name in names_of_joints_to_remove]
for jr in joints_to_remove:
    modified_urdf._joints.remove(jr)

# Add virtual base joint (only translation, no rotation)
joint_base_translation = urdfpy.Joint(
    name='joint_base_translation',
    parent='base_link',
    child='link_base_translation',
    joint_type='prismatic',
    axis=np.array([1.0, 0.0, 0.0]),
    origin=np.eye(4, dtype=np.float64),
    limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0),
)
modified_urdf._joints.append(joint_base_translation)
link_base_translation = urdfpy.Link(name='link_base_translation', inertial=None, visuals=None, collisions=None)
modified_urdf._links.append(link_base_translation)

for j in modified_urdf._joints:
    if j.name == 'joint_mast':
        j.parent = 'link_base_translation'

new_urdf_path = "/tmp/iktutorial/stretch.urdf"
modified_urdf.save(new_urdf_path)

chain = ikpy.chain.Chain.from_urdf_file(new_urdf_path)
for link in chain.links:
    print(f"* Link Name: {link.name}, Type: {link.joint_type}")


def get_current_configuration():
    def bound_range(name, value):
        names = [l.name for l in chain.links]
        if name not in names:
            return value
        index = names.index(name)
        bounds = chain.links[index].bounds
        return min(max(value, bounds[0]), bounds[1])

    q_base = bound_range('joint_base_translation', robot.base.status['x'])   # 底盘 x 供 IK；执行时转为增量
    q_lift = bound_range('joint_lift', robot.lift.status['pos'])
    q_arml = bound_range('joint_arm_l0', robot.arm.status['pos'] / 4.0)
    q_yaw = bound_range('joint_wrist_yaw', robot.end_of_arm.status['wrist_yaw']['pos'])
    q_pitch = bound_range('joint_wrist_pitch', robot.end_of_arm.status['wrist_pitch']['pos'])
    q_roll = bound_range('joint_wrist_roll', robot.end_of_arm.status['wrist_roll']['pos'])
    return [0.0, q_base, 0.0, q_lift, 0.0, q_arml, q_arml, q_arml, q_arml, q_yaw, 0.0, q_pitch, q_roll, 0.0, 0.0]


def move_to_configuration(q):
    # Stretch 底盘只有 translate_by(delta) 相对位移，无 move_to。IK 的 q[1] 是绝对 x，需转为增量
    desired_base_x = q[1]
    current_base_x = robot.base.status['x']
    delta_x = desired_base_x - current_base_x
    q_lift = q[3]
    q_arm = q[5] + q[6] + q[7] + q[8]
    q_yaw = q[9]
    q_pitch = q[11]
    q_roll = q[12]
    robot.base.translate_by(delta_x)
    robot.lift.move_to(q_lift)
    robot.arm.move_to(q_arm)
    robot.end_of_arm.move_to('wrist_yaw', q_yaw)
    robot.end_of_arm.move_to('wrist_pitch', q_pitch)
    robot.end_of_arm.move_to('wrist_roll', q_roll)
    robot.push_command()


def _clamp_to_chain_bounds(q):
    """将关节角限制在 chain 各关节限位内。"""
    q = np.array(q, dtype=np.float64)
    for i in range(min(len(q), len(chain.links))):
        lo, hi = chain.links[i].bounds
        q[i] = np.clip(q[i], lo, hi)
    return q


def move_to_grasp_goal(target_point, target_orientation_matrix):
    """target_orientation_matrix: 3x3 旋转矩阵。orientation_mode='Z' 时 ikpy 需要 (3,) 的 Z 轴向量。"""
    q_init = get_current_configuration()
    # orientation_mode='Z' 时传末端 Z 轴方向向量 (3,)，不能传 3x3
    z_axis = np.array(target_orientation_matrix)[:3, 2]
    q_soln = chain.inverse_kinematics(target_point, z_axis, orientation_mode='Z', initial_position=q_init)
    print('Solution:', q_soln)
    q_use = _clamp_to_chain_bounds(q_soln)
    err = np.linalg.norm(chain.forward_kinematics(q_use)[:3, 3] - np.array(target_point))
    ERR_LIMIT = 0.5   # 50 cm 内按可行解执行
    print("钳制后位置误差 %.4f m（阈值 %.1f m）" % (err, ERR_LIMIT))
    if err > ERR_LIMIT:
        print("IKPy did not find a valid solution（钳制后仍超 50 cm）")
        print("提示：从顶端降下来请用负的 z（相对顶端偏移），例如 z=-0.05 表示降 5 cm；z 为正表示比顶端还高，可能不可达。")
        return None
    if np.any(np.abs(np.array(q_use) - np.array(q_soln)) > 1e-6):
        print("已按关节限位钳制后执行。")
    print("执行动作中...")
    move_to_configuration(q=q_use)
    return q_use


def get_current_grasp_pose():
    q = get_current_configuration()
    return chain.forward_kinematics(q)


# ---------- 启动：top = lift(max - 0.18)，记录此时末端 z 为 top_z；之后目标位置均为 (x, y, top_z + z) ----------
print("正在升到 top = lift(max - 0.18)...")
_lift_lo, _lift_hi = robot.lift.soft_motion_limits['hard']
robot.lift.move_to(_lift_hi - 0.18)
robot.push_command()
robot.wait_command()
_top_pose = get_current_grasp_pose()
TOP_Z = float(_top_pose[2, 3])   # top_z
print("top_z = %.4f m；之后目标位置均为 (x, y, top_z + z)。" % TOP_Z)

# ---------- 死循环：目标位置 (x, y, top_z + z)，输入 x y z qx qy qz qw ----------
print("每轮先打印当前位姿，再输入 action: x y z qx qy qz qw（目标 z = top_z + z）。Ctrl+C 退出。")
while True:
    print("当前位姿 (4x4):")
    print(get_current_grasp_pose())
    s = input("action > ").strip()
    try:
        vals = [float(x) for x in s.split()]
        if len(vals) != 7:
            print("需要恰好 7 个数字 (x y z qx qy qz qw)，请重试。")
            continue
        x, y, z, qx, qy, qz, qw = vals
        x, y, z = 0.01 * x, 0.01 * y, 0.01 * z   # 输入尺度 x100，转为米
        target_z = TOP_Z + z
        if target_z > TOP_Z:
            target_z = TOP_Z
            print("目标 z 已限制为 top_z（输入 z>0 时不再高于顶端）")
        target_point = [x, y, target_z]   # 目标位置 = (x, y, top_z + z)，且 z 不超过 top_z
        target_orientation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        move_to_grasp_goal(target_point, target_orientation)
    except ValueError as e:
        print("解析失败，请输入 7 个数字，用空格分隔。", e)
    except Exception as e:
        print("执行出错:", e)
