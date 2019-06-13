import sys, os, cv2, copy
sys.path.append("/home/mqm/Documents/computer/darknet/python")  # So darknet.py is in PYTHONPATH
# It's also necessary to edit darknet.py with the path to libdarknet.so
#print sys.path

import darknet as dn
import pdb

#
# Detect pegs
#
dn.set_gpu(0)  # Use the first GPU
net = dn.load_net("/home/mqm/Documents/computer/darknet/cfg/yolo-empty-test.cfg", "/home/mqm/Documents/computer/darknet/yolo-empty_19250.weights", 0)
meta = dn.load_meta("/home/mqm/Documents/computer/darknet/cfg/empty.data")
image_string = "/home/mqm/Documents/computer/darknet/data/photos_lab/c8.png"
r = dn.detect(net, meta, image_string, thresh = 0.38)
print r


#
# Draw bounding boxes around detected pegs and show image
#
pic = cv2.imread(image_string)
for i in range(0, len(r)):  # 640 x 480
        if(r[i][0] == 'near'):
                color = (0, 255, 0)                                
                print("Certainty percentage (near): " + str(r[i][1]*100))
        
        if(r[i][0] == 'far'):
                color = (0, 0, 255)
                print("Certainty percentage (far): " + str(r[i][1]*100))

        cv2.rectangle(pic, (int(r[i][2][0])-int(r[i][2][2]/2), int(r[i][2][1])-int(r[i][2][3]/2)), (int(r[i][2][0])+int(r[i][2][2]/2), int(r[i][2][1])+int(r[i][2][3]/2)), color, 3)

cv2.imshow('image', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
