import csv
import math
import numpy as np
import re
import socket
import sys
import time
import zmq

# Transformation to tool flange from camera
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
T_AT = np.concatenate((np.concatenate((R_TA.T, -R_TA.T @ P_A_T), axis=1), np.array([[0, 0, 0, 1]])))

# Poses and joint vectors
HOME_CONFIG_I = [math.radians(x) for x in [0, -45, -135, -180, -90, 0]]
HOME_CONFIG_II = [math.radians(x) for x in [-180, -135, 135, -90, 45, -90]]
    
SCAN_CONFIG_I = [math.radians(x) for x in [8, -102, -107, -151, -103, -183]]
TOP_SCAN_I = np.array([.910, -.174, .922, 3.27, -2.20, 2.06])
PLACE_CONFIG_I = [[math.radians(x) for x in [16, -73, -86, -11, 8, -192]],
                  [math.radians(x) for x in [16, -71, -124, 26, 8, -192]],
                  [math.radians(x) for x in [16, -99, -144, 72, 9, -191]],
                  [math.radians(x) for x in [13, -116, -148, -75, -13, -20]],
                  [math.radians(x) for x in [13, -166, -132, -43, -13, -19]],
                  [math.radians(x) for x in [13, -198, -101, -45, -14, -18]]]

SCAN_CONFIG_II = [math.radians(x) for x in [-170, -77, 102, -24, 131, 3]]
TOP_SCAN_II = np.array([.858, .352, 1.015, 1.45, .77, .84])
PLACE_CONFIG_II = [[math.radians(x) for x in [-181, -109, 91, -164, -20, 2]],
                   [math.radians(x) for x in [-180, -109, 126, -199, -20, 2]],
                   [math.radians(x) for x in [-180, -82, 144, -244, -20, 3]],
                  [math.radians(x) for x in [-180, -30, 139, -290, -19, 3]],
                  [math.radians(x) for x in [-182, -18, 136, -121, 17, -178]],
                  [math.radians(x) for x in [-182, 16, 103, -122, 17, -178]]]

# Misc. constants
D_PEG_SPACING_Z = .325  # Distance between pegs in base z-direction
CAMERA_X_FUDGE_FACTOR = -.007#.015  # m difference between actual x-direction tag pose and reported value
CAMERA_Y_FUDGE_FACTOR = -.013#.015
CAMERA_Z_FUDGE_FACTOR = .035#.015#.035

JOINT_SPEEDS_NORM_THRESHOLD = .01  # Max allowable norm of joint speeds (in rad/s)
global_theta = 0  # Using global variable when converting Euler axis to/from rotation matrix


def get_T_BP():
    T_BA = get_robot_pose()

    #
    # Calculate transformation to camera frame from peg frame
    # For now assume only one tag found
    #
    poses_list = get_tag_poses()
    print("poses found:")
    print(poses_list)
    R_CP = np.array(poses_list[1:10]).reshape(3,3)
    P_P_C = np.array(poses_list[10:13]).reshape(3,1)
    P_P_C[0,0] = P_P_C[0,0] + CAMERA_X_FUDGE_FACTOR
    P_P_C[1,0] = P_P_C[1,0] + CAMERA_Y_FUDGE_FACTOR
    P_P_C[2,0] = P_P_C[2,0] + CAMERA_Z_FUDGE_FACTOR
    T_CP = np.concatenate((np.concatenate((R_CP, P_P_C), axis=1), np.array([[0, 0, 0, 1]])))

    T_BP = T_BA @ T_AT @ T_TC @ T_CP
    
    return T_BP


def get_peg_poses(quadrant):
    ret = []
    
    if 1 == quadrant:
        # Get poses for quadrant-I pegs
        #
        send_robot_msg(SCAN_CONFIG_I, "1")  # MoveJ to nominal quadrant-I peg-scan configuration in joint coordinates
        wait_for_robot_to_stop()
        
        for x in range(6):  # Estimate pose of all six quadrant-I pegs
            target_pose = np.array(TOP_SCAN_I)
            target_pose[2] = target_pose[2] - x * D_PEG_SPACING_Z
            send_robot_msg(target_pose, "3")  # MoveL to pose for taking picture of peg
            wait_for_robot_to_stop()
            time.sleep(1)  # Let robot quit vibrating
            ret.append(get_T_BP())
            
        target_pose = np.array(TOP_SCAN_I)
        target_pose[2] = target_pose[2] - 2 * D_PEG_SPACING_Z
        send_robot_msg(target_pose, "3")
        wait_for_robot_to_stop()
        send_robot_msg(HOME_CONFIG_I, "1")
        wait_for_robot_to_stop()

    elif 2 == quadrant:
        # Get poses for quadrant-II pegs
        #
        send_robot_msg(SCAN_CONFIG_II, "1")  # MoveJ to nominal quadrant-II peg-scan configuration in joint coordinates
        wait_for_robot_to_stop()
        
        for x in range(6):  # Estimate pose of all six quadrant-II pegs
            target_pose = np.array(TOP_SCAN_II)
            target_pose[2] = target_pose[2] - x * D_PEG_SPACING_Z
            send_robot_msg(target_pose, "3")  # MoveL to pose for taking picture of peg
            wait_for_robot_to_stop()
            time.sleep(1)  # Let robot quit vibrating
            ret.append(get_T_BP())
            
        # Return robot to nice configuration
        target_pose = np.array(TOP_SCAN_II)
        target_pose[2] = target_pose[2] - 2 * D_PEG_SPACING_Z
        send_robot_msg(target_pose, "3")  # MoveL to pose for taking picture of peg
        wait_for_robot_to_stop()
        send_robot_msg(HOME_CONFIG_II, "1")  # MoveJ to start config
        wait_for_robot_to_stop()

    print(ret)

    return ret


def test_place(T_BP, quadrant):
    # Constants for placement
    D_FORK_INSERTION = .105
    D_TUBE_LENGTH = .29
    D_PEG_LIP = .113
    TESTING_SAFETY_DISTANCE = 0
    D_CENTER_AXIS = .01
    D_PEG_RIB_OFFSET = .0015
    D_APPROACH_CLEARANCE = 0
    D_LATERAL_MOVE = .18
    D_PLACE_DROP = .015  # After releasing spool, move this distance in base -z-direction
    
    R_PA_goal = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]])
    P_A_P_a1 = np.array([[D_PEG_LIP+D_APPROACH_CLEARANCE+D_TUBE_LENGTH-D_FORK_INSERTION+TESTING_SAFETY_DISTANCE, -D_CENTER_AXIS, D_PEG_RIB_OFFSET-D_LATERAL_MOVE]]).T
    P_A_P_a2 = np.array([[D_PEG_LIP+D_APPROACH_CLEARANCE+D_TUBE_LENGTH-D_FORK_INSERTION+TESTING_SAFETY_DISTANCE, -D_CENTER_AXIS, D_PEG_RIB_OFFSET]]).T
    P_A_P_place = np.array([[D_PEG_LIP-D_FORK_INSERTION+TESTING_SAFETY_DISTANCE, -D_CENTER_AXIS, D_PEG_RIB_OFFSET]]).T

    if 2 == quadrant:
        R_PA_goal = np.identity(3)
        P_A_P_a1[0] = -P_A_P_a1[0]
        P_A_P_a2[0] = -P_A_P_a2[0]
        P_A_P_place[0] = -P_A_P_place[0]

    #
    # Move onto the peg
    #
    T_PA_a1 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a1), axis=1), np.array([[0, 0, 0, 1]])))
    T_PA_a2 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a2), axis=1), np.array([[0, 0, 0, 1]])))
    T_PA_place = np.concatenate((np.concatenate((R_PA_goal, P_A_P_place), axis=1), np.array([[0, 0, 0, 1]])))
    
    T_BA_a1 = T_BP @ T_PA_a1
    T_BA_a2 = T_BP @ T_PA_a2
    T_BA_place = T_BP @ T_PA_place
    
    m = calc_axis_angle(T_BA_a1[0:3, 0:3])
    a1_pose = np.array([T_BA_a1[0,3], T_BA_a1[1,3], T_BA_a1[2,3], m[0], m[1], m[2]])
    send_robot_msg(a1_pose, "3")  # MoveL
    wait_for_robot_to_stop()
    
    m = calc_axis_angle(T_BA_a2[0:3, 0:3])
    a2_pose = np.array([T_BA_a2[0,3], T_BA_a2[1,3], T_BA_a2[2,3], m[0], m[1], m[2]])
    send_robot_msg(a2_pose, "3")  # MoveL
    wait_for_robot_to_stop()
    
    m = calc_axis_angle(T_BA_place[0:3, 0:3])
    place_pose = np.array([T_BA_place[0,3], T_BA_place[1,3], T_BA_place[2,3], m[0], m[1], m[2]])
    send_robot_msg(place_pose, "3")  # MoveL
    wait_for_robot_to_stop()

    #
    # Reverse course
    #
    is_placing = False
    if is_placing:
        send_robot_msg([0], "")  # Release spool
        wait_for_robot_to_stop()

        P_A_P_place = np.array([[D_PEG_LIP-D_FORK_INSERTION+TESTING_FUDGE_DISTANCE, -D_CENTER_AXIS+D_PLACE_DROP, D_PEG_RIB_OFFSET]]).T
        P_A_P_a2 = np.array([[D_PEG_LIP+D_APPROACH_CLEARANCE+D_TUBE_LENGTH-D_FORK_INSERTION+TESTING_FUDGE_DISTANCE, -D_CENTER_AXIS+D_PLACE_DROP, D_PEG_RIB_OFFSET]]).T
        P_A_P_a1 = np.array([[D_PEG_LIP+D_APPROACH_CLEARANCE+D_TUBE_LENGTH-D_FORK_INSERTION+TESTING_FUDGE_DISTANCE, -D_CENTER_AXIS+D_PLACE_DROP, D_PEG_RIB_OFFSET-D_LATERAL_MOVE]]).T
        
        if 2 == quadrant:
            P_A_P_place[0] = -P_A_P_place[0]
            P_A_P_a2[0] = -P_A_P_a2[0]
            P_A_P_a1[0] = -P_A_P_a1[0]
        
        T_PA_place = np.concatenate((np.concatenate((R_PA_goal, P_A_P_place), axis=1), np.array([[0, 0, 0, 1]])))
        T_PA_a2 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a2), axis=1), np.array([[0, 0, 0, 1]])))
        T_PA_a1 = np.concatenate((np.concatenate((R_PA_goal, P_A_P_a1), axis=1), np.array([[0, 0, 0, 1]])))
        
        T_BA_place = T_BP @ T_PA_place
        T_BA_a2 = T_BP @ T_PA_a2
        T_BA_a1 = T_BP @ T_PA_a1
        
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
        
    else:
        send_robot_msg(a2_pose, "3")  # MoveL
        wait_for_robot_to_stop()
        send_robot_msg(a1_pose, "3")  # MoveL
        wait_for_robot_to_stop()
        
    
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


# Returns homogenous transformation matrix to robot base from robot TCP
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


# Ensure robot has stopped moving
# It seems this function is uncessary because the robot will not send string
# until it has stopped motion command
def wait_for_robot_to_stop():
    joint_speeds = np.array(get_robot_joint_speeds())
    print(np.linalg.norm(joint_speeds))
    while (JOINT_SPEEDS_NORM_THRESHOLD < np.linalg.norm(joint_speeds)):
        joint_speeds = np.array(get_robot_joint_speeds())
        print(np.linalg.norm(joint_speeds))


#
# The message convention is that c can be either 1, 2, or 3
# If 6 == len(x) then c corresponds to 1:movej in joint coords, 2:movej in pose coords, 3:movel
# If 1 == len(x) then c corresponds to 1:pick or 2:place
#
def send_robot_msg(x, c):
    string_for_robot = ','.join(str(e) for e in x)  # Surely a terrible method for sending list of floats
    string_for_robot = "(" + string_for_robot + "," + c + ")"
    print("Sending this message to robot:\n" + string_for_robot + "\n")
    #robot_connection.send(b"(10)")
    robot_connection.send(string_for_robot.encode())


if __name__ == "__main__":
    # Connect to server providing machine vision for peg poses
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
    for x in range(4):  # Flush old messages?
        get_robot_pose()
        
    '''
    #
    # Testing --- debug
    #
    send_robot_msg(TEMP_CONFIG, "1")  # MoveJ to start config
    wait_for_robot_to_stop()
    time.sleep(1)
    T_BP = get_T_BP()

    # Calc pose in front of tag
    R_PA_touch = np.array([[0, 0, -1],
                          [0, 1, 0],
                          [1, 0, 0]])
    P_A_P_touch = np.array([[0, 0, -.05]]).T    
    T_PA_touch = np.concatenate((np.concatenate((R_PA_touch, P_A_P_touch), axis=1), np.array([[0, 0, 0, 1]])))    
    T_BA_touch = T_BP @ T_PA_touch
    m = calc_axis_angle(T_BA_touch[0:3, 0:3])
    touch_pose = np.array([T_BA_touch[0,3], T_BA_touch[1,3], T_BA_touch[2,3], m[0], m[1], m[2]])

    # Calc pose approaching from side of tag
    R_PA_touch = np.identity(3)
    P_A_P_touch = np.array([[-.2, 0, 0]]).T
    T_PA_touch = np.concatenate((np.concatenate((R_PA_touch, P_A_P_touch), axis=1), np.array([[0, 0, 0, 1]])))    
    T_BA_touch = T_BP @ T_PA_touch
    m = calc_axis_angle(T_BA_touch[0:3, 0:3])
    touch_pose = np.array([T_BA_touch[0,3], T_BA_touch[1,3], T_BA_touch[2,3], m[0], m[1], m[2]])
    send_robot_msg(touch_pose, "3")  # MoveL
    wait_for_robot_to_stop()
    
    P_A_P_touch = np.array([[-.05, 0, 0]]).T
    T_PA_touch = np.concatenate((np.concatenate((R_PA_touch, P_A_P_touch), axis=1), np.array([[0, 0, 0, 1]])))    
    T_BA_touch = T_BP @ T_PA_touch
    m = calc_axis_angle(T_BA_touch[0:3, 0:3])
    touch_pose = np.array([T_BA_touch[0,3], T_BA_touch[1,3], T_BA_touch[2,3], m[0], m[1], m[2]])

    send_robot_msg(touch_pose, "3")  # MoveL
    wait_for_robot_to_stop()
    
    sys.exit()
    '''
    quadrant = 1
    if 1 == quadrant:
        send_robot_msg(HOME_CONFIG_I, "1")  # MoveJ to start config
    elif 2 == quadrant:
        send_robot_msg(HOME_CONFIG_II, "1")  # MoveJ to start config
    wait_for_robot_to_stop()

    poses = get_peg_poses(quadrant)
    print(poses)
    durations = []
    
    # Go through pseudo place motion
    for x in range(6):  # Quadrant-1 pegs
        start_time = time.time()
        send_robot_msg(PLACE_CONFIG_I[x], "1")
        wait_for_robot_to_stop()
        test_place(poses[x], quadrant)
        send_robot_msg(PLACE_CONFIG_I[x], "1")
        wait_for_robot_to_stop()
        send_robot_msg(HOME_CONFIG_I, "1")  # MoveJ to start config
        wait_for_robot_to_stop()
        end_time = time.time()
        durations.append(end_time - start_time)
    
    for x in range(0):  # Quadrant-2 pegs
        start_time = time.time()
        send_robot_msg(PLACE_CONFIG_II[x], "1")
        wait_for_robot_to_stop()
        test_place(poses[x], quadrant)
        send_robot_msg(PLACE_CONFIG_II[x], "1")
        wait_for_robot_to_stop()
        send_robot_msg(HOME_CONFIG_II, "1")  # MoveJ to start config
        wait_for_robot_to_stop()
        end_time = time.time()
        durations.append(end_time - start_time)

    # Write info to file
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    print(poses)
    print(durations)
    sys.stdout = orig_stdout
    f.close()
    
    close_connections()
