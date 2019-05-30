import sys, os
sys.path.append("/home/mqm/Documents/computer/darknet/python")  # So darknet.py is in PYTHONPATH
# It's also necessary to edit darknet.py with the path to libdarknet.so
print sys.path

import darknet as dn

import pdb

dn.set_gpu(0)  # This is for using the first GPU?  That is, it's different from "nogpu"?
net = dn.load_net("/home/mqm/Documents/computer/darknet/cfg/yolo-empty-test.cfg", "/home/mqm/Documents/computer/darknet/yolo-empty_19250.weights", 0)
meta = dn.load_meta("/home/mqm/Documents/computer/darknet/cfg/empty.data")
r = dn.detect(net, meta, "/home/mqm/Documents/computer/darknet/data/photos_lab/c1.png")

print r
