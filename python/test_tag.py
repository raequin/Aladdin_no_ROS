import math
import numpy as np
import re
import socket
import sys
import time
import zmq



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

    close_connections()
        
