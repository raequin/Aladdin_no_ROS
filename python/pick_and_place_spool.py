import math
import numpy as np
import re
import socket
import sys
import time
import zmq


# "Constants"
IMAGE_LEFT = 0
IMAGE_WIDTH = 640
IMAGE_TOP = 0
IMAGE_HEIGHT = 512
IMAGE_CENTER_U = IMAGE_WIDTH / 2
IMAGE_CENTER_V = IMAGE_HEIGHT / 2
CIRCLE_ERROR_TOLERANCE = 8  # Acceptable image error norm, in pixels
CIRCLE_RADIUS_MIN = 35
CIRCLE_RADIUS_MAX = 85
PROPORTIONAL_VS_GAIN = 1/2000  # Convert error in pixels to m (sort of)
CAMERA_X_FUDGE_FACTOR = .015  # m difference between actual x-direction tag pose and reported value
CAMERA_Y_FUDGE_FACTOR = .015
CAMERA_Z_FUDGE_FACTOR = .015#.035
PICK_MOVE = [.372, -.052, .04]

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
T_AT = np.concatenate((np.concatenate((R_TA.T, -R_TA.T @ P_A_T), axis=1), np.array([[0, 0, 0, 1]])))

PEG_SCAN_POSE = np.array([-.750, -.026, .56, .871, -1.48, -1.64])  # Pose for camera viewing pegs
PEG_SCAN_JOINT = [math.radians(x) for x in [11, -84, -113, -153, -134, -172]]
SPOOL_SCAN_POSE = np.array([-.56, .3, -.01, 2.2214, 0.0, -2.2214])  # Pose for camera over first spool
INTERMEDIATE_POSE = np.array([.371, .245, .419, 4.16, -.85, -.73])  # Pose between peg_scan and spool_scan
BASE_SPOOL_Z = -.56390  # z-coordinate of bottom layer of spools
SPOOL_OFFSET_Z = .31204  # z-direction distance between layers of spools
INTERMEDIATE_JOINT = [math.radians(x) for x in [108, -47, -111, -220, -54, -90]]
HOME_JOINT = [math.radians(x) for x in [-180, -90, -90, -180, -90, -90]]
OVER_JOINT = [math.radians(x) for x in [70, -23, -119, -200, -98, -161]]

#PICK_TRANSLATION = .375  # Distance (in m) for tool to move along spool axis
#PLACE_TRANSLATION = .1  # Distance (in m) for tool to move along peg axis
JOINT_SPEEDS_NORM_THRESHOLD = .01  # Max allowable norm of joint speeds (in rad/s)

global_theta = 0


def test_tag(T_CP, T_BA):
    '''
    R_PA_goal = np.array([[0, 0, -1],
                     [0, 1, 0],
                     [1, 0, 0]])    
    P_A_P_goal = np.array([[0, 0, -.1]]).T
    '''
    R_PA_goal = np.identity(3)
    P_A_P_goal = np.array([[-.1, 0, 0]]).T
    
    T_PA_goal = np.concatenate((np.concatenate((R_PA_goal, P_A_P_goal), axis=1), np.array([[0, 0, 0, 1]])))

    T_BA_goal = T_BA @ T_AT @ T_TC @ T_CP @ T_PA_goal
    print(T_BA_goal)

    m = calc_axis_angle(T_BA_goal[0:3, 0:3])
    test_pose = np.array([T_BA_goal[0,3], T_BA_goal[1,3], T_BA_goal[2,3], m[0], m[1], m[2]])

    send_robot_msg(test_pose, "3")


def test_place(T_CP, T_BA_image):
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

    T_BA_a1 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a1
    T_BA_a2 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a2
    T_BA_place = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_place

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

    T_BA_place = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_place
    T_BA_a2 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a2
    T_BA_a1 = T_BA_image @ T_AT @ T_TC @ T_CP @ T_PA_a1
    
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
'''
def calc_axis_angle(R):
    c = (R[0,0]+R[1,1]+R[2,2]-1) / 2
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

    return [mx, my, mz]
''' 

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


#
# Returns 0 if no target for servoing
# Returns 1 once target reached
#
def visual_servoing():
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
                robot_error_vec_T = (R_TC @ image_error) * PROPORTIONAL_VS_GAIN
                robot_error_vec_B = T_BA[0:3, 0:3] @ robot_error_vec_T
                print("Error vector in robot base frame:")
                print(robot_error_vec_B)
                desired_robot_pose = [T_BA[0,3]+robot_error_vec_B[0,0], T_BA[1,3]+robot_error_vec_B[1,0], SPOOL_SCAN_POSE[2], SPOOL_SCAN_POSE[3], SPOOL_SCAN_POSE[4], SPOOL_SCAN_POSE[5]]
#                desired_robot_pose = [curr_pose[0]+robot_error_vec_B[0,0], curr_pose[1]+robot_error_vec_B[1,0], SPOOL_SCAN_POSE[2], SPOOL_SCAN_POSE[3], SPOOL_SCAN_POSE[4], SPOOL_SCAN_POSE[5]]
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

    #
    # For now assume only one tag found
    #
    R_CP = np.array(poses_list[0:9]).reshape(3,3)
    P_P_C = np.array(poses_list[9:12]).reshape(3,1)
    P_P_C[0,0] = P_P_C[0,0] + CAMERA_X_FUDGE_FACTOR
    P_P_C[1,0] = P_P_C[1,0] + CAMERA_Y_FUDGE_FACTOR
    P_P_C[2,0] = P_P_C[2,0] + CAMERA_Z_FUDGE_FACTOR
    T_CP = np.concatenate((np.concatenate((R_CP, P_P_C), axis=1), np.array([[0, 0, 0, 1]])))
    print(T_CP)

    print("movel to INTERMEDIATE_POSE")
    send_robot_msg(INTERMEDIATE_POSE, "3")
    wait_for_robot_to_stop()

    print("movej to SPOOL_SCAN_POSE")
    send_robot_msg(SPOOL_SCAN_POSE, "2")
    wait_for_robot_to_stop()

    print("entering visual_servoing")
    visual_servoing_result = visual_servoing()
    print("visual_servoing_result = {0}".format(visual_servoing_result))

    if 1 == visual_servoing_result:
        #input("Visual servoing is complete.  Press Enter to continue")
        send_robot_msg(PICK_MOVE, "1")  # Initiate pick routine
    else:
        print("No target for visual servoing")
        sys.exit()
        

    POST_PICK_JOINT_01 = [math.radians(x) for x in [153, -106, -119, -135, -165, -90]]
    POST_PICK_JOINT_02 = [math.radians(x) for x in [153, -35, -143, -135, -165, -80]]
    POST_PICK_JOINT_03 = [math.radians(x) for x in [24, -35, -143, -135, -165, -80]]
    POST_PICK_JOINT_04 = [math.radians(x) for x in [24, -35, -143, -135, -30, -80]]
    POST_PICK_JOINT_05 = [math.radians(x) for x in [24, -76, -107, -135, -30, -80]]
    
#    send_robot_msg(POST_PICK_JOINT_01, "1")
#    wait_for_robot_to_stop()
    send_robot_msg(POST_PICK_JOINT_02, "1")
    wait_for_robot_to_stop()
    send_robot_msg(POST_PICK_JOINT_03, "1")
    wait_for_robot_to_stop()
    send_robot_msg(POST_PICK_JOINT_04, "1")
    wait_for_robot_to_stop()
    send_robot_msg(POST_PICK_JOINT_05, "1")
    wait_for_robot_to_stop()

#    print("move to PEG_SCAN")
#    send_robot_msg(PEG_SCAN_JOINT, "1")  # Movej to home pose in joint coordinates
#    wait_for_robot_to_stop()
    #test_tag(T_CP, T_BA)
    test_place(T_CP, T_BA)

    send_robot_msg(POST_PICK_JOINT_05, "1")
    wait_for_robot_to_stop()
    
    close_connections()
        
    #time.sleep(1)
