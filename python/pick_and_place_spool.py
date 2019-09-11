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

# Values used by peg-scan routine
PEG_SCAN_POSE = np.array([-.750, -.026, .56, .871, -1.48, -1.64])  # Pose for camera viewing pegs
PEG_SCAN_JOINT = [math.radians(x) for x in [5, -81, -103, -169, -129, -175]]  # Pose for camera viewing pegs
INTERMEDIATE_JOINT = [math.radians(x) for x in [-16, -47, -116, -197, -90, -90]]  # Pose between peg_scan and home

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
PROPORTIONAL_VS_GAIN = .0005  # Convert error in pixels to m (sort of)

# Values used by cone-pick routine
# Looking down on pallet, first cone to be picked is at origin of pallet coordinate system
NUM_ROWS = 1  # Count of cones in pallet y-direction
NUM_COLS = 1  # Count of cones in pallet x-direction
NUM_LAYERS = 1  # How many layers of cones on the pallet
CONE_SPACING = .29  # Distance from cone center to cone center (m)
PALLET_LAYER_HEIGHT = .32  # Vertical distance from top of one cone layer to next
BOTTOM_CONE_HEIGHT = -.541  # z-coordinate of cone-tube top in robot base frame
CONE_TOP_TO_TOOL_DIST_FOR_SCAN = .491  # Offset in cone-axis direction between cone top and tool flange during scan
CONE_TOP_TO_TOOL_DIST_FOR_PICK = .159  # Offset in cone-axis direction between cone top and tool flange when actuator grasps cone
NOMINAL_FIRST_CONE_SCAN_X = -.712  # x-coordinate (in base frame) for scanning nominal first cone location
NOMINAL_FIRST_CONE_SCAN_Y = .271  # y-coordinate (in base frame) for scanning nominal first cone location
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


#
# Methods
#
global_theta = 0  # Using a global variable for some reason (laziness?)

def calc_axis_angle(R):
    # Method from "Kinematic Analysis of Robot Manipulators," by Duffy and Crane
    '''
    print("R")
    print(R)
    c = (R[0,0]+R[1,1]+R[2,2]-1) / 2
    print("c = {0}".format(c))
    if math.pi < global_theta or 0 > global_theta:
        c = -c
    print("c = {0}".format(c))

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

    print("mx, my, mz, theta")
    print(mx)
    print(my)
    print(mz)
    print(theta)
    sys.exit()
    
    return [mx*theta, my*theta, mz*theta]
    '''
    # Method from Wikipedia page for rotation matrix
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]

    return [rx, ry, rz]


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


#
# Because at this time, ease of development is deemed to be more valuable than performance,
# get the path to an image saved on the drive instead of using shared memory  :-/
#
def get_image_path():
    print("Sending request for image")
    vision_socket.send(b"Request image")
    msg = vision_socket.recv()
    ret = msg.decode("utf-8")
    
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


def get_robot_pose():  # Returns homogenous transformation matrix to robot base from robot TCP
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
    
    print("curr_pose, P_A_B, curr_r")
    print(curr_pose)
    print(P_A_B)
    print(curr_r)
    print("global_theta = {0}".format(global_theta))
    print("R_BA then T_BA")
    print(R_BA)
    print(T_BA)
    #sys.exit()
    
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
                print("R_TC then image_error")
                print(R_TC)
                print(image_error)
                robot_error_vec_T = np.matmul(R_TC, image_error) * PROPORTIONAL_VS_GAIN   # Desired robot offset in tool frame
                robot_error_vec_B = np.matmul(T_BA[0:3, 0:3], robot_error_vec_T)  # Desired robot offset in base frame
                print("Error vector in robot tool frame:")
                print(robot_error_vec_T)
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


def place_a_cone(T_CP, T_BA_image):
    print("T_CP, T_BA_image")
    print(T_CP)
    print(T_BA_image)
    
    R_PA_goal = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]])

    d_fork_insertion = .105
    d_tube_length = .29
    d_peg_lip = .113
    testing_fudge_distance = 0
    d_center_axis = .01
    d_peg_rib_offset = .0015
    d_approach_clearance = .025
    d_lateral_move = .18
    d_place_drop = .015  # After releasing spool, move this distance in base -z-direction
    
    P_A_P_a1 = np.array([[d_peg_lip+d_approach_clearance+d_tube_length-d_fork_insertion+testing_fudge_distance, -d_center_axis, d_peg_rib_offset-d_lateral_move]]).T
    P_A_P_a2 = np.array([[d_peg_lip+d_approach_clearance+d_tube_length-d_fork_insertion+testing_fudge_distance, -d_center_axis, d_peg_rib_offset]]).T
    P_A_P_place = np.array([[d_peg_lip-d_fork_insertion+testing_fudge_distance, -d_center_axis, d_peg_rib_offset]]).T
    
    T_PA_a1 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a1), axis=1), np.array([[0, 0, 0, 1]])))
    T_PA_a2 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a2), axis=1), np.array([[0, 0, 0, 1]])))
    T_PA_place = np.concatenate((np.concatenate((R_PA_goal, P_A_P_place), axis=1), np.array([[0, 0, 0, 1]])))

    #T_BA_a1 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a1
    #T_BA_a2 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a2
    #T_BA_place = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_place
    T_BA_a1 = np.matmul(T_BA_image, np.matmul(T_AT, np.matmul(T_TC, np.matmul(T_CP, T_PA_a1))))
    T_BA_a2 = np.matmul(T_BA_image, np.matmul(T_AT, np.matmul(T_TC, np.matmul(T_CP, T_PA_a2))))
    T_BA_place = np.matmul(T_BA_image, np.matmul(T_AT, np.matmul(T_TC, np.matmul(T_CP, T_PA_place))))

    print("T_BA_a1, T_BA_a2, T_BA_place")
    print(T_BA_a1)
    print(T_BA_a2)
    print(T_BA_place)

    m = calc_axis_angle(T_BA_a1[0:3, 0:3])
    a1_pose = np.array([T_BA_a1[0,3], T_BA_a1[1,3], T_BA_a1[2,3], m[0], m[1], m[2]])
    send_robot_msg(a1_pose, "2")  # MoveJ
    wait_for_robot_to_stop()
    
    m = calc_axis_angle(T_BA_a2[0:3, 0:3])
    a2_pose = np.array([T_BA_a2[0,3], T_BA_a2[1,3], T_BA_a2[2,3], m[0], m[1], m[2]])
    send_robot_msg(a2_pose, "3")  # MoveL
    wait_for_robot_to_stop()
    
    m = calc_axis_angle(T_BA_place[0:3, 0:3])
    place_pose = np.array([T_BA_place[0,3], T_BA_place[1,3], T_BA_place[2,3], m[0], m[1], m[2]])
    send_robot_msg(place_pose, "3")  # MoveL
    wait_for_robot_to_stop()
    '''
    send_robot_msg(a2_pose, "3")
    wait_for_robot_to_stop()
    send_robot_msg(a1_pose, "3")
    wait_for_robot_to_stop()
    '''
    send_robot_msg([0], "")  # Release spool
    wait_for_robot_to_stop()

    # Move in base -z-direction then retract actuator
    P_A_P_place = np.array([[d_peg_lip-d_fork_insertion+testing_fudge_distance, -d_center_axis+d_place_drop, d_peg_rib_offset]]).T
    P_A_P_a2 = np.array([[d_peg_lip+d_approach_clearance+d_tube_length-d_fork_insertion+testing_fudge_distance, -d_center_axis+d_place_drop, d_peg_rib_offset]]).T
    P_A_P_a1 = np.array([[d_peg_lip+d_approach_clearance+d_tube_length-d_fork_insertion+testing_fudge_distance, -d_center_axis+d_place_drop, d_peg_rib_offset-d_lateral_move]]).T
    
    T_PA_place = np.concatenate((np.concatenate((R_PA_goal, P_A_P_place), axis=1), np.array([[0, 0, 0, 1]])))
    T_PA_a2 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a2), axis=1), np.array([[0, 0, 0, 1]])))
    T_PA_a1 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a1), axis=1), np.array([[0, 0, 0, 1]])))

    #T_BA_place = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_place
    #T_BA_a2 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a2
    #T_BA_a1 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a1
    T_BA_place = np.matmul(T_BA_image, np.matmul(T_AT, np.matmul(T_TC, np.matmul(T_CP, T_PA_place))))
    T_BA_a2 = np.matmul(T_BA_image, np.matmul(T_AT, np.matmul(T_TC, np.matmul(T_CP, T_PA_a2))))
    T_BA_a1 = np.matmul(T_BA_image, np.matmul(T_AT, np.matmul(T_TC, np.matmul(T_CP, T_PA_a1))))
    
    m = calc_axis_angle(T_BA_place[0:3, 0:3])
    place_pose = np.array([T_BA_place[0,3], T_BA_place[1,3], T_BA_place[2,3], m[0], m[1], m[2]])
    send_robot_msg(place_pose, "3")  # MoveL
    wait_for_robot_to_stop()
    
    m = calc_axis_angle(T_BA_a2[0:3, 0:3])
    a2_pose = np.array([T_BA_a2[0,3], T_BA_a2[1,3], T_BA_a2[2,3], m[0], m[1], m[2]])
    send_robot_msg(a2_pose, "3")  # MoveL
    wait_for_robot_to_stop()

    m = calc_axis_angle(T_BA_a1[0:3, 0:3])
    a1_pose = np.array([T_BA_a1[0,3], T_BA_a1[1,3], T_BA_a1[2,3], m[0], m[1], m[2]])
    send_robot_msg(a1_pose, "3")  # MoveL
    wait_for_robot_to_stop()


#
#  main
#
if __name__ == "__main__":
    HOME_JOINT = [math.radians(x) for x in [-90, -47, -116, -197, -90, -90]]
    #HOME_JOINT = [math.radians(x) for x in [-90, -90, -90, -180, -90, -90]]

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

    print("movej to INTERMEDIATE_JOINT")
    send_robot_msg(INTERMEDIATE_JOINT, "1")
    wait_for_robot_to_stop()

    # Find tag pose
    print("move to PEG_SCAN")
#    send_robot_msg(PEG_SCAN_POSE, "3")
    send_robot_msg(PEG_SCAN_JOINT, "1")  # Movej to home pose in joint coordinates
    wait_for_robot_to_stop()

    T_BA = get_robot_pose()  # Get current transformation to base from TCP
    print(T_BA)
    print("getting peg poses")
    poses_list = get_tag_poses()
    print("poses found:")
    print(poses_list)

    # For now assume only one tag found
    R_CP = np.array(poses_list[0:9]).reshape(3,3)
    P_P_C = np.array(poses_list[9:12]).reshape(3,1)
    P_P_C[0,0] = P_P_C[0,0] + CAMERA_X_FUDGE_FACTOR
    P_P_C[1,0] = P_P_C[1,0] + CAMERA_Y_FUDGE_FACTOR
    P_P_C[2,0] = P_P_C[2,0] + CAMERA_Z_FUDGE_FACTOR
    T_CP = np.concatenate((np.concatenate((R_CP, P_P_C), axis=1), np.array([[0, 0, 0, 1]])))
    print("T_CP")
    print(T_CP)

    # Done getting tag info, return to home pose
    '''
    print("movej to INTERMEDIATE_JOINT")
    send_robot_msg(INTERMEDIATE_JOINT, "1")
    wait_for_robot_to_stop()

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

    # Hang a cone
    print("movej to HOME_JOINT")
    send_robot_msg(HOME_JOINT, "1")
    wait_for_robot_to_stop()
    '''
    print("movej to INTERMEDIATE_JOINT")
    send_robot_msg(INTERMEDIATE_JOINT, "1")
    wait_for_robot_to_stop()

    place_a_cone(T_CP, T_BA)
    
    close_connections()
