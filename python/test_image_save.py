import math
import numpy as np
import re
import socket
import sys
import time
import zmq



def close_connections():
    vision_context.destroy()


def get_image_path():
    vision_socket.send(b"Request image")
    msg = vision_socket.recv()
    ret = msg.decode("utf-8")
    
    return ret




if __name__ == "__main__":
    # Connect to server providing machine vision for peg poses and spool circles
    print('Connecting to machine-vision server')
    vision_context = zmq.Context()
    vision_socket = vision_context.socket(zmq.REQ)
    vision_socket.connect("tcp://127.0.0.1:43001")

    image_path = get_image_path()  # Ask machine-vision server for spool (image) coordinates
    print(image_path)
    close_connections()
