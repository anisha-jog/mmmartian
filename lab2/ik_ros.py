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
    # 仅保留绝对值逻辑：Stretch 底盘只有 translate_by(delta)，用 目标x - 当前x 得到增量
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
    """target_point: (x,y,z)。姿态固定为夹爪朝下，用完整 3x3 矩阵 + orientation_mode='all'。"""
    q_init = get_current_configuration()
    # 夹爪朝下且朝右：base 中 Y 左、Z 上，故“下右”为 (0, 1, -1) 单位化；末端接近轴取为该方向
    _inv_sqrt2 = 1.0 / np.sqrt(2.0)
    approach = np.array([0.0, _inv_sqrt2, -_inv_sqrt2])   # 下右
    forward = np.array([1.0, 0.0, 0.0])                    # 前
    right = np.cross(approach, forward)
    right = right / np.linalg.norm(right)
    forward = np.cross(right, approach)
    gripper_down_rotation = np.column_stack((approach, right, forward)).astype(np.float64)
    q_soln = chain.inverse_kinematics(target_point, gripper_down_rotation, orientation_mode='all', initial_position=q_init)
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


def pose_4x4_to_8(pose_4x4):
    """将 4x4 位姿矩阵转为 8 维向量：x, y, z, qx, qy, qz, qw, 1。"""
    pose_4x4 = np.asarray(pose_4x4)
    x, y, z = pose_4x4[0, 3], pose_4x4[1, 3], pose_4x4[2, 3]
    R = pose_4x4[:3, :3]
    quat = Rotation.from_matrix(R).as_quat()  # xyzw
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    return [x, y, z, qx, qy, qz, qw, 1.0]


# ---------- Panda (Franka Emika 7-DoF) / LIBERO → Stretch 参数转换 ----------
# Franka Panda: base 原点，X 前 Y 左 Z 上；名义 workspace 中心 [0.515, 0, 0.226] m，约 0.4^3 盒
# LIBERO 动作：7 维 (x,y,z, qx,qy,qz,qw) 或 (x,y,z, roll,pitch,yaw, gripper)，单位米
# Stretch: 同 convention，但臂长/行程不同，用线性缩放+偏移映射到 Stretch 可达范围
PANDA_X_CENTER = 0.515   # Panda 名义 workspace 中心 x (m)
PANDA_Y_CENTER = 0.0
PANDA_Z_CENTER = 0.226   # 桌面高度典型值
STRETCH_X_CENTER = 0.25  # Stretch 前方手臂舒适中心 x
STRETCH_Y_CENTER = 0.0
PANDA_TO_STRETCH_SCALE_X = 0.8   # Panda 方向 x 缩放
PANDA_TO_STRETCH_SCALE_Y = 0.8
PANDA_TO_STRETCH_SCALE_Z = 1.0   # Panda z 相对桌面 → Stretch z 缩放
STRETCH_Z_REF = 0.5               # Stretch 目标 z 的固定参考高度 (m)，不再用 top
ARM_EXTEND_OFFSET = 0.29
Z_OFFSET_CM = 0.6                # 所有目标 z 加高


def panda_to_stretch_position(x_panda, y_panda, z_panda):
    """将 Panda/LIBERO 空间末端位置 (米) 转为 Stretch base_link 下目标 (x,y,z)。不再用 top_z。"""
    x_s = (x_panda - PANDA_X_CENTER) * PANDA_TO_STRETCH_SCALE_X + STRETCH_X_CENTER + ARM_EXTEND_OFFSET
    y_s = (y_panda - PANDA_Y_CENTER) * PANDA_TO_STRETCH_SCALE_Y + STRETCH_Y_CENTER - ARM_EXTEND_OFFSET
    z_s = STRETCH_Z_REF + (z_panda - PANDA_Z_CENTER) * PANDA_TO_STRETCH_SCALE_Z + Z_OFFSET_CM
    return (x_s, y_s, z_s)

# ---------- 启动：z_top = Stretch 末端可达最高点（lift 上限时正解 z），只复位一次到 (0, 0, z_top) ----------
_lift_lo, _lift_hi = robot.lift.soft_motion_limits['hard']
_q_max = get_current_configuration()
# 配置里第 4 个为 joint_lift（与 get_current_configuration 顺序一致）
_q_max[3] = _lift_hi
Z_TOP = float(chain.forward_kinematics(np.array(_q_max, dtype=np.float64))[2, 3])
print("z_top = %.4f m（Stretch 末端最高点）；复位到 (0, 0, z_top)...")
move_to_grasp_goal([0.0, 0.0, Z_TOP], np.eye(3))
robot.wait_command()
print("已复位，之后每轮只输入 action。")

# ---------- 死循环：输入 Panda 空间 x y z qx qy qz qw 并执行 ----------
print("每轮先打印当前位姿，再输入 action: x y z qx qy qz qw（Panda/LIBERO 单位：米）。Ctrl+C 退出。")
while True:
    pose_4x4 = get_current_grasp_pose()
    pose_8 = pose_4x4_to_8(pose_4x4)
    print("当前位姿 (8 位):", ",".join("%g" % v for v in pose_8))
    s = input("action > ").strip()
    try:
        vals = [float(x) for x in s.split()]
        if len(vals) != 7:
            print("需要恰好 7 个数字 (x y z qx qy qz qw)，请重试。")
            continue
        x_p, y_p, z_p, qx, qy, qz, qw = vals
        x_p, y_p, z_p = 0.1 * x_p, 0.1 * y_p, 0.1 * z_p   # 输入尺度 x10，转为米
        target_point = list(panda_to_stretch_position(x_p, y_p, z_p))
        target_orientation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        move_to_grasp_goal(target_point, target_orientation)
    except ValueError as e:
        print("解析失败，请输入 7 个数字，用空格分隔。", e)
    except Exception as e:
        print("执行出错:", e)
