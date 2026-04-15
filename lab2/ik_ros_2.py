import csv
import os
import time
import urchin as urdfpy
import numpy as np
import ikpy.chain
import stretch_body.robot
import importlib.resources as importlib_resources
from scipy.spatial.transform import Rotation
# NOTE: `python3 -m pip install --upgrade ikpy graphviz urchin networkx scipy`

# ---------- Stretch Python API (no ROS) ----------
robot = stretch_body.robot.Robot()
robot.startup()
if not robot.is_calibrated():
    robot.home()
# Stow to a safe tucked pose first
robot.stow()
robot.push_command()
robot.wait_command()

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

    q_base = bound_range('joint_base_translation', robot.base.status['x'])   # Base x for IK; executed as delta
    q_lift = bound_range('joint_lift', robot.lift.status['pos'])
    q_arml = bound_range('joint_arm_l0', robot.arm.status['pos'] / 4.0)
    q_yaw = bound_range('joint_wrist_yaw', robot.end_of_arm.status['wrist_yaw']['pos'])
    q_pitch = bound_range('joint_wrist_pitch', robot.end_of_arm.status['wrist_pitch']['pos'])
    q_roll = bound_range('joint_wrist_roll', robot.end_of_arm.status['wrist_roll']['pos'])
    return [0.0, q_base, 0.0, q_lift, 0.0, q_arml, q_arml, q_arml, q_arml, q_yaw, 0.0, q_pitch, q_roll, 0.0, 0.0]


def move_to_configuration(q):
    # Absolute targets: base uses translate_by(delta) only, so delta_x = desired_x - current_x
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
    """Clamp joint values to each link's bounds in the chain."""
    q = np.array(q, dtype=np.float64)
    for i in range(min(len(q), len(chain.links))):
        lo, hi = chain.links[i].bounds
        q[i] = np.clip(q[i], lo, hi)
    return q


def move_to_grasp_goal(target_point, target_orientation_matrix, add_table_delta=True):
    """target_point: (x,y,z). Gripper pose is fixed (down); uses full 3x3 + orientation_mode='all'."""
    tp = np.asarray(target_point, dtype=np.float64).copy()
    if add_table_delta:
        tp[2] += DELTA_TABLE
    target_point = tp
    q_init = get_current_configuration()
    # Gripper down and to the right: in base frame Y left, Z up, so "down-right" is (0, 1, -1) normalized
    _inv_sqrt2 = 1.0 / np.sqrt(2.0)
    approach = np.array([0.0, _inv_sqrt2, -_inv_sqrt2])   # down-right
    forward = np.array([1.0, 0.0, 0.0])                    # forward
    right = np.cross(approach, forward)
    right = right / np.linalg.norm(right)
    forward = np.cross(right, approach)
    gripper_down_rotation = np.column_stack((approach, right, forward)).astype(np.float64)
    q_soln = chain.inverse_kinematics(target_point, gripper_down_rotation, orientation_mode='all', initial_position=q_init)
    print('Solution:', q_soln)
    q_use = _clamp_to_chain_bounds(q_soln)
    err = np.linalg.norm(chain.forward_kinematics(q_use)[:3, 3] - np.array(target_point))
    ERR_LIMIT = 0.5   # Accept solution if within 50 cm after clamping
    print("Position error after clamp: %.4f m (limit %.1f m)" % (err, ERR_LIMIT))
    if err > ERR_LIMIT:
        print("IKPy did not find a valid solution (still > 50 cm after clamping)")
        print("Hint: to move down from the top use negative z (offset from top), e.g. z=-0.05 is 5 cm lower; positive z is higher and may be unreachable.")
        return None
    if np.any(np.abs(np.array(q_use) - np.array(q_soln)) > 1e-6):
        print("Executing after clamping to joint limits.")
    print("Executing motion...")
    move_to_configuration(q=q_use)
    return q_use


def get_current_grasp_pose():
    q = get_current_configuration()
    return chain.forward_kinematics(q)


def pose_4x4_to_8(pose_4x4):
    """Convert 4x4 pose to 8-vector: x, y, z, qx, qy, qz, qw, 1."""
    pose_4x4 = np.asarray(pose_4x4)
    x, y, z = pose_4x4[0, 3], pose_4x4[1, 3], pose_4x4[2, 3]
    R = pose_4x4[:3, :3]
    quat = Rotation.from_matrix(R).as_quat()  # xyzw
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    return [x, y, z, qx, qy, qz, qw, 1.0]


# ---------- Panda (Franka 7-DoF) / LIBERO -> Stretch mapping ----------
# Franka Panda: base origin, X forward Y left Z up; nominal workspace center [0.515, 0, 0.226] m
# LIBERO action: 7D (x,y,z, qx,qy,qz,qw) or (x,y,z, r,p,y, gripper), meters
# Stretch: same convention; scale and offset map into Stretch reach
PANDA_X_CENTER = 0.515   # Panda nominal workspace center x (m)
PANDA_Y_CENTER = 0.0
PANDA_Z_CENTER = 0.226   # Typical table height
STRETCH_X_CENTER = 0.25  # Comfortable arm center in front of Stretch
STRETCH_Y_CENTER = 0.0
PANDA_TO_STRETCH_SCALE_X = 0.8
PANDA_TO_STRETCH_SCALE_Y = 0.8
PANDA_TO_STRETCH_SCALE_Z = 1.0
STRETCH_Z_REF = 0.5               # Fixed reference height for Stretch z (m)
ARM_EXTEND_OFFSET = 0.29
Z_OFFSET_CM = 0.42                # Extra z offset in mapping (m)
#DELTA_TABLE = 0.26                # Added to goal z (m) in move_to_grasp_goal when add_table_delta=True
DELTA_TABLE = 0                # Added to goal z (m) in move_to_grasp_goal when add_table_delta=True

# CSV / execute_panda_pose_action xyz interpretation:
# - True (default): x,y,z are Panda/LIBERO frame (m); panda_to_stretch_position() subtracts PANDA_*_CENTER and maps to Stretch.
# - False: x,y,z are already absolute end-effector position in Stretch base_link (m); no center/offset remap (quaternion unchanged).
ACTION_XYZ_IS_STRETCH_BASE = False

# If True: run all rows from ACTION_CSV_PATH (or ACTION_ROWS) once, then exit. If False: interactive input loop.
ACT_DIRECTLY = True
ACTION_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ik_ros_2_actions.csv")
# Fallback when CSV is missing: list of [x, y, z, qx, qy, qz, qw] in Panda/LIBERO meters (same as spreadsheet columns).
ACTION_ROWS = []


def panda_to_stretch_position(x_panda, y_panda, z_panda):
    """Map Panda/LIBERO end-effector position (m) to Stretch base_link target (x,y,z)."""
    x_s = (x_panda - PANDA_X_CENTER) * PANDA_TO_STRETCH_SCALE_X + STRETCH_X_CENTER + ARM_EXTEND_OFFSET
    y_s = (y_panda - PANDA_Y_CENTER) * PANDA_TO_STRETCH_SCALE_Y + STRETCH_Y_CENTER - ARM_EXTEND_OFFSET
    z_s = STRETCH_Z_REF + (z_panda - PANDA_Z_CENTER) * PANDA_TO_STRETCH_SCALE_Z + Z_OFFSET_CM
    return (x_s, y_s, z_s)


def load_actions_from_csv(path):
    """Load 7 floats per row (x,y,z,qx,qy,qz,qw). Skips header / bad lines.

    Excel paste often uses TABs between columns; csv.reader only splits on commas,
    so tabbed rows become one cell and were silently dropped (IndexError).
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
            else:
                parts = [p.strip() for p in next(csv.reader([line])) if p.strip()]
            if len(parts) < 7:
                continue
            try:
                vals = [float(parts[i]) for i in range(7)]
            except ValueError:
                continue
            rows.append(vals)
    return rows


def execute_panda_pose_action(x_p, y_p, z_p, qx, qy, qz, qw, *, xyz_scale):
    """Map Panda pose to Stretch and execute. xyz_scale=0.1 for interactive x10 typing; 1.0 for table values in meters."""
    x_p, y_p, z_p = xyz_scale * x_p, xyz_scale * y_p, xyz_scale * z_p
    if ACTION_XYZ_IS_STRETCH_BASE:
        target_point = [float(x_p), float(y_p), float(z_p)]
    else:
        target_point = list(panda_to_stretch_position(x_p, y_p, z_p))
    target_orientation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    move_to_grasp_goal(target_point, target_orientation, add_table_delta=True)


# ---------- Startup: z_top = max EE height (FK at lift hard limit); home once to (0, 0, z_top) ----------
_lift_lo, _lift_hi = robot.lift.soft_motion_limits['hard']
_q_max = get_current_configuration()
# Index 3 is joint_lift (same order as get_current_configuration)
_q_max[3] = _lift_hi
Z_TOP = float(chain.forward_kinematics(np.array(_q_max, dtype=np.float64))[2, 3])
print("z_top = %.4f m (Stretch EE max height); homing to (0, 0, z_top)..." % Z_TOP)
move_to_grasp_goal([0.0, 0.0, Z_TOP], np.eye(3), add_table_delta=False)
robot.wait_command()
print("Homed.")
time.sleep(2)

if ACT_DIRECTLY:
    if os.path.isfile(ACTION_CSV_PATH):
        direct_rows = load_actions_from_csv(ACTION_CSV_PATH)
    else:
        direct_rows = [list(r) for r in ACTION_ROWS]
    if not direct_rows:
        raise SystemExit(
            "ACT_DIRECTLY=True but no actions: export your table to %s (7 numeric columns per row) "
            "or set ACTION_ROWS in ik_ros_2.py." % ACTION_CSV_PATH
        )
    print("Running %d direct actions (table xyz in meters, z + %.3f m in move_to_grasp_goal)..." % (len(direct_rows), DELTA_TABLE))
    for i, row in enumerate(direct_rows):
        x_p, y_p, z_p, qx, qy, qz, qw = row
        print("Direct step %d/%d:" % (i + 1, len(direct_rows)), row)
        try:
            execute_panda_pose_action(x_p, y_p, z_p, qx, qy, qz, qw, xyz_scale=1.0)
            robot.wait_command()
            time.sleep(2)
        except Exception as e:
            print("Execution error at row %d:" % (i + 1), e)
else:
    print("Each round: current pose is printed, then enter action: x y z qx qy qz qw (Panda/LIBERO, meters). Ctrl+C to exit.")
    while True:
        pose_4x4 = get_current_grasp_pose()
        pose_8 = pose_4x4_to_8(pose_4x4)
        print("Current pose (8):", ",".join("%g" % v for v in pose_8))
        s = input("action > ").strip()
        try:
            vals = [float(x) for x in s.split()]
            if len(vals) != 7:
                print("Need exactly 7 numbers (x y z qx qy qz qw); try again.")
                continue
            x_p, y_p, z_p, qx, qy, qz, qw = vals
            execute_panda_pose_action(x_p, y_p, z_p, qx, qy, qz, qw, xyz_scale=0.1)
            robot.wait_command()
            time.sleep(2)
        except ValueError as e:
            print("Parse error; enter 7 space-separated numbers.", e)
        except Exception as e:
            print("Execution error:", e)
