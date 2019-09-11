import math
import numpy as np
import re
import socket
import sys
import time
import zmq


# AprilTag estimate seems to be off.  Here are translation corrections.
CAMERA_X_FUDGE_FACTOR = 0#.015  # m difference between actual x-direction tag pose and reported value
CAMERA_Y_FUDGE_FACTOR = 0#.015
CAMERA_Z_FUDGE_FACTOR = 0#.015



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


def get_peg_poses():
    print("Sending request for poses")
    vision_socket.send(b"Request peg poses")
    msg = vision_socket.recv()
    ret = bytes_to_list(msg)
    
    return ret


if __name__ == "__main__":
    # Connect to server providing machine vision for peg poses and spool circles
    print('Connecting to machine-vision server')
    vision_context = zmq.Context()
    vision_socket = vision_context.socket(zmq.REQ)
    vision_socket.connect("tcp://127.0.0.1:43001")


    print("getting peg poses")
    while 1==1:
        poses_list = get_peg_poses()
        print("poses found:")
        print(poses_list)

        # For now assume one tag found
        R_CP = np.array(poses_list[1:10]).reshape(3,3)
        P_P_C = np.array(poses_list[10:13]).reshape(3,1)
        P_P_C[0,0] = P_P_C[0,0] + CAMERA_X_FUDGE_FACTOR
        P_P_C[1,0] = P_P_C[1,0] + CAMERA_Y_FUDGE_FACTOR
        P_P_C[2,0] = P_P_C[2,0] + CAMERA_Z_FUDGE_FACTOR
        T_CP = np.concatenate((np.concatenate((R_CP, P_P_C), axis=1), np.array([[0, 0, 0, 1]])))
        print(T_CP)


    close_connections()
        
