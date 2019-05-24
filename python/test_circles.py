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


def get_spool_circles():
    vision_socket.send(b"Request spool circles")
    msg = vision_socket.recv()
    ret = bytes_to_list(msg)
    
    return ret




if __name__ == "__main__":
    # Connect to server providing machine vision for peg poses and spool circles
    print('Connecting to machine-vision server')
    vision_context = zmq.Context()
    vision_socket = vision_context.socket(zmq.REQ)
    vision_socket.connect("tcp://127.0.0.1:43001")

    while 1==1:        
        circles_list = get_spool_circles()  # Ask machine-vision server for spool (image) coordinates
        print(circles_list)        
    close_connections()
        
    #time.sleep(1)
