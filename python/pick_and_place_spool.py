import math
import numpy as np
import re
import socket
import sys
import time
import zmq

#
# "Constants"
#

# Transformation from camera to tool flange
R_TC = np.array([[0, 0, 1],
                 [0, 1, 0],
                 [-1, 0, 0]])  # Rotation matrix to camera frame from tool flange
P_C_T = np.array([[.025, -.063, .154]]).T
T_TC = np.concatenate((np.concatenate((R_TC, P_C_T), axis=1), np.array([[0, 0, 0, 1]])))

# Transformation to tool flange from actuator
R_TA = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
P_A_T = np.array([[.205, 0, .112]]).T
T_TA = np.concatenate((np.concatenate((R_TA, P_A_T), axis=1), np.array([[0, 0, 0, 1]])))

# Transformation to actuator from tool flange
T_AT = np.concatenate((np.concatenate((R_TA.T, np.matmul(-R_TA.T, P_A_T)), axis=1), np.array([[0, 0, 0, 1]])))

# Values used by cone-pick machine vision
IMAGE_LEFT = 0
IMAGE_WIDTH = 640
IMAGE_TOP = 0
IMAGE_HEIGHT = 512
IMAGE_CENTER_U = IMAGE_WIDTH / 2
IMAGE_CENTER_V = IMAGE_HEIGHT / 2
CIRCLE_ERROR_TOLERANCE = 4  # Acceptable image error norm, in pixels
CIRCLE_RADIUS_MIN = 55
CIRCLE_RADIUS_MAX = 95
PROPORTIONAL_VS_GAIN = 1/2000  # Convert error in pixels to m (sort of)

# Values used by cone-pick routine
# Looking down on pallet, first cone to be picked is at origin of pallet coordinate system
NUM_ROWS = 4  # Count of cones in pallet y-direction
NUM_COLS = 2  # Count of cones in pallet x-direction
NUM_LAYERS = 1  # How many layers of cones on the pallet
CONE_SPACING = .29  # Distance from cone center to cone center (m)
PALLET_LAYER_HEIGHT = .32  # Vertical distance from top of one cone layer to next
BOTTOM_CONE_HEIGHT = -.541  # z-coordinate of cone-tube top in robot base frame
CONE_TOP_TO_TOOL_DIST_FOR_SCAN = .491  # Offset in cone-axis direction between cone top and tool flange during scan
CONE_TOP_TO_TOOL_DIST_FOR_PICK = .159  # Offset in cone-axis direction between cone top and tool flange when actuator grasps cone
NOMINAL_FIRST_CONE_SCAN_X = -.712  # x-coordinate (in base frame) for scanning nominal first cone location
NOMINAL_FIRST_CONE_SCAN_Y = .371  # y-coordinate (in base frame) for scanning nominal first cone location
NOMINAL_FIRST_CONE_SCAN_Z = BOTTOM_CONE_HEIGHT + CONE_TOP_TO_TOOL_DIST_FOR_SCAN + (NUM_LAYERS-1) * PALLET_LAYER_HEIGHT  # z-coordinate (in base frame) for scanning nominal first cone location
NOMINAL_FIRST_CONE_SCAN_POSE = np.array([NOMINAL_FIRST_CONE_SCAN_X, NOMINAL_FIRST_CONE_SCAN_Y, NOMINAL_FIRST_CONE_SCAN_Z, 2.2214, 0.0, -2.2214])  # Pose for camera over first spool
NOMINAL_PALLET_THETA = math.pi  # Angle from robot-base x direction to pallet x direction (row direction)
PICK_MOVE = [CONE_TOP_TO_TOOL_DIST_FOR_SCAN - CONE_TOP_TO_TOOL_DIST_FOR_PICK,
             (P_C_T[1]-P_A_T[1])[0],
             (P_C_T[2]-P_A_T[2])[0]]  # Movement (in tool frame) from spool-centered-in-camera to grasping spool

CONE_TUBE_LENGTH = .286  # Length of cardboard cone tube, only used for this check MQM 190527
if CONE_TUBE_LENGTH > PICK_MOVE[0]:
    print("Let's make the pick scanning height be at least one cone length over pallet, shall we?")
    sys.exit()

# AprilTag estimate seems to be off.  Here are translation corrections.
CAMERA_X_FUDGE_FACTOR = .015  # m difference between actual x-direction tag pose and reported value
CAMERA_Y_FUDGE_FACTOR = .015
CAMERA_Z_FUDGE_FACTOR = .015

# Max allowable norm of joint speeds (in rad/s) to be considered stopped (not really useful, see wait_for_robot_to_stop())
JOINT_SPEEDS_NORM_THRESHOLD = .01

global_theta = 0  # Using a global variable for some reason (laziness?)


def calc_axis_angle(R):
    c = (R[0,0]+R[1,1]+R[2,2]-1) / 2
    if math.pi < global_theta or 0 > global_theta:
        c = -c
        
    if 0 < R[2,1]-R[1,2]:
        smx = 1
    else:
        smx = -1
    if 0 < R[0,2]-R[2,0]:
        smy = 1
    else:
        smy = -1
    if 0 < R[1,0]-R[0,1]:
        smz = 1
    else:
        smz = -1
        
    mx = smx * math.sqrt((R[0,0]-c) / (1 - c))
    my = smy * math.sqrt((R[1,1]-c) / (1 - c))
    mz = smz * math.sqrt((R[2,2]-c) / (1 - c))

    theta = math.acos(c)

    return [mx*theta, my*theta, mz*theta]


def close_connections():
    robot_connection.close()
    robot_socket.close()
    vision_context.destroy()


def bytes_to_list(msg):
    msg_str = msg.decode("utf-8")
    print("Message received:\n" + msg_str)
    msg_str_values = re.findall(r"[-+]?\d*\.\d+|\d+", msg_str)
    msg_values = [float(x) for x in msg_str_values]

    return msg_values


def get_spool_circles():
    vision_socket.send(b"Request spool circles")
    msg = vision_socket.recv()
    ret = bytes_to_list(msg)
    
    return ret


def get_tag_poses():
    print("Sending request for poses")
    vision_socket.send(b"Request peg poses")
    msg = vision_socket.recv()
    ret = bytes_to_list(msg)
    
    return ret


def receive_robot_message(is_waiting_for_pose):
    msg = robot_connection.recv(1024)
    msg_str = msg.decode("utf-8")
    print("Message received from robot:\n" + msg_str)
    if is_waiting_for_pose:  # A pose value will have a leading "p"
        msg_str_values = msg_str[2:-1].split(",")
        
    else:
        msg_str_values = msg_str[1:-1].split(",")
        
    ret = [float(x) for x in msg_str_values]
    print(ret)

    return ret


#
# Returns homogenous transformation matrix to robot base from robot TCP
#
def get_robot_pose():
    send_robot_msg([1], "1")  # Send two ones to robot for pose request
    print("Awaiting robot coordinates")

    curr_pose = receive_robot_message(True)
    P_A_B = np.array([curr_pose[0:3]]).T
    curr_r = np.array(curr_pose[3:6])
    global_theta = np.linalg.norm(curr_r)
    m = curr_r / global_theta
    c = math.cos(global_theta)
    s = math.sin(global_theta)
    v = 1 - c
    R_BA = np.array([[m[0]*m[0]*v+c, m[0]*m[1]*v-m[2]*s, m[0]*m[2]*v+m[1]*s],
                    [m[0]*m[1]*v+m[2]*s, m[1]*m[1]*v+c, m[1]*m[2]*v-m[0]*s],
                    [m[0]*m[2]*v-m[1]*s, m[1]*m[2]*v+m[0]*s, m[2]*m[2]*v+c]])
    T_BA = np.concatenate((np.concatenate((R_BA, P_A_B), axis=1), np.array([[0, 0, 0, 1]])))
    
    return T_BA


def get_robot_joint_speeds():
    send_robot_msg([2], "2")  # Send two twos to robot for joint-speed request
    print("Awaiting robot joint speeds")

    return receive_robot_message(False)


#
# Ensure robot has stopped moving
# It seems this function is uncessary because the robot will not send string
# until it has stopped motion command
#
def wait_for_robot_to_stop():
    joint_speeds = np.array(get_robot_joint_speeds())
    print(np.linalg.norm(joint_speeds))
    while (JOINT_SPEEDS_NORM_THRESHOLD < np.linalg.norm(joint_speeds)):
        joint_speeds = np.array(get_robot_joint_speeds())
        print(np.linalg.norm(joint_speeds))


#
# The message convention is that c can be either 1, 2, or 3
# NOTE: THE FOLLOWING DESCRIPTION NEEDS TO BE UPDATED (MQM 190527)
# If 6 == len(x) then c corresponds to 1:movej in joint coords, 2:movej in pose coords, 3:movel
# If 1 == len(x) then c corresponds to 1:pick or 2:place
#
def send_robot_msg(x, c):
    string_for_robot = ','.join(str(e) for e in x)  # Surely a terrible method for sending list of floats
    string_for_robot = "(" + string_for_robot + "," + c + ")"
    print("Sending this message to robot:\n" + string_for_robot + "\n")
    robot_connection.send(string_for_robot.encode())


#
# Returns 0 if no target for servoing
# Returns 1 once target reached
#
def visual_servoing(this_cone_scan_pose):
    while True:
        #input("Press Enter to continue")
        wait_for_robot_to_stop()
        T_BA = get_robot_pose()  # Get current transformation to base from TCP
        circles_list = get_spool_circles()  # Ask machine-vision server for spool (image) coordinates
        print("{0} circles found".format(len(circles_list) / 3))
        print(circles_list)
        
        # If circle(s) found, then send position offset to robot for central circle
        if 3 <= len(circles_list):
            
            # Find circle closest to center of image
            min_error_norm = float('inf')
            central_circle_index = 0            
            for x in range (0, len(circles_list)-1, 3):
                this_error_u = circles_list[x] - IMAGE_CENTER_U
                this_error_v = circles_list[x + 1] - IMAGE_CENTER_V
                this_error_norm = math.sqrt(math.pow(this_error_u, 2) + math.pow(this_error_v, 2))
                
                if min_error_norm > this_error_norm and CIRCLE_RADIUS_MAX > circles_list[x + 2] and CIRCLE_RADIUS_MIN < circles_list[x + 2]:  # Also check that circle is a spool tube
                    min_error_norm = this_error_norm
                    central_circle_index = x
                
            # If central circle center not close enough to image center, then send offset to robot, otherwise finished
            # Create error vector in image plane and force it to be a column vector
            image_error = np.array([circles_list[central_circle_index] - IMAGE_CENTER_U, circles_list[central_circle_index + 1] - IMAGE_CENTER_V, 0])[:, np.newaxis]
            print("error_u = {0}\terror_v = {1}\tfor circle {2}\n".format(image_error[0], image_error[1], central_circle_index/3))
            if CIRCLE_ERROR_TOLERANCE < np.linalg.norm(image_error):
                robot_error_vec_T = np.matmul(R_TC, image_error) * PROPORTIONAL_VS_GAIN  # Desired robot offset in tool frame
                robot_error_vec_B = np.matmul(T_BA[0:3, 0:3], robot_error_vec_T)  # Desired robot offset in base frame
                print("Error vector in robot base frame:")
                print(robot_error_vec_B)
                desired_robot_pose = [T_BA[0,3]+robot_error_vec_B[0,0], T_BA[1,3]+robot_error_vec_B[1,0], this_cone_scan_pose[2],
                                      this_cone_scan_pose[3], this_cone_scan_pose[4], this_cone_scan_pose[5]]
                send_robot_msg(desired_robot_pose, "3")

            else:
                return 1

            #
            # NOTE: Work yet to be done using radii for approximating distance and rejecting
            # erroneous matches
            #

        else:            
            print("No circles found\n")
            return 0


if __name__ == "__main__":
    HOME_JOINT = [math.radians(x) for x in [-180, -90, -90, -180, -90, -90]]

    # Connect to server providing machine vision for peg poses and spool circles
    print('Connecting to machine-vision server')
    vision_context = zmq.Context()
    vision_socket = vision_context.socket(zmq.REQ)
    vision_socket.connect("tcp://127.0.0.1:43001")

    # Create server for UR 10 and wait for connection
    print("Creating socket and listening for robot")
    HOST = "192.168.1.104"
    PORT = 43000
    robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    robot_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    robot_socket.bind((HOST, PORT))
    robot_socket.listen(5)
    robot_connection, addr = robot_socket.accept()
    print('Robot connected to this server')
    for x in range(6):  # Flush old messages?
        get_robot_pose()

    print("movej to HOME_JOINT")
    send_robot_msg(HOME_JOINT, "1")
    wait_for_robot_to_stop()

    # Pick all the cones
    for i in range(NUM_LAYERS):
        for j in range(NUM_COLS):
            for k in range(NUM_ROWS):
                # Calculate offsets in pallet directions
                this_vertical_offset = -PALLET_LAYER_HEIGHT * i
                this_row_dir_offset = CONE_SPACING * j
                this_col_dir_offset = CONE_SPACING * k

                # Create pose for starting VS routine for current cone
                this_scan_pose = np.copy(NOMINAL_FIRST_CONE_SCAN_POSE)
                this_scan_pose[0] += math.cos(NOMINAL_PALLET_THETA) * this_row_dir_offset + math.sin(NOMINAL_PALLET_THETA) * this_col_dir_offset
                this_scan_pose[1] += math.cos(NOMINAL_PALLET_THETA) * this_col_dir_offset + math.sin(NOMINAL_PALLET_THETA) * this_row_dir_offset
                this_scan_pose[2] += this_vertical_offset

                # Move the robot to scan pose (start VS pose)
                print("movej to offset scan pose")
                send_robot_msg(this_scan_pose, "2")
                wait_for_robot_to_stop()

                # Visual servoing to get cone axis and camera optical axis coincident
                print("entering visual_servoing")
                visual_servoing_result = visual_servoing(NOMINAL_FIRST_CONE_SCAN_POSE)
                print("visual_servoing_result = {0}".format(visual_servoing_result))
                
                if 1 == visual_servoing_result:
                    #input("Visual servoing is complete.  Press Enter to continue")
                    send_robot_msg(PICK_MOVE, "1")  # Initiate pick routine
                    wait_for_robot_to_stop()
                    send_robot_msg(HOME_JOINT, "1")
                    wait_for_robot_to_stop()
                else:
                    print("No target for visual servoing")
                    sys.exit()

    
    close_connections()
