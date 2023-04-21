#!/usr/bin/env python3
#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 05 - LINE AND PLANE SEGMENTATION BY RANSAC AND PCA
#
import numpy
import cv2
import ros_numpy
import rospy
import math
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from vision_msgs.srv import FindLines
from vision_msgs.srv import FindPlanes
from vision_msgs.srv import RecognizeObjects
from visualization_msgs.msg import MarkerArray

NAME = "FULL_NAME"

def filter_image(cloud):
    global bgr
    xyz  = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud)
    mask = (xyz[:,0] < 0.3)# | (xyz[:,0] > 2.0)# | (xyz[:,1] < -2.0) | (xyz[:,1] > 2.0) | (xyz[:,2] < 0.3) | (xyz[:,2] > 2.0)
    mask = numpy.reshape(mask, (480,640))
    rgb  = ros_numpy.point_cloud2.pointcloud2_to_array(cloud)['rgb'].copy()
    rgb.dtype = numpy.uint32
    r,g,b = ((rgb >> 16) & 255), ((rgb >> 8) & 255), (rgb & 255)
    bgr = cv2.merge((numpy.asarray(b,dtype='uint8'),numpy.asarray(g,dtype='uint8'),numpy.asarray(r,dtype='uint8')))
    bgr[mask]=(0,0,0)

def callback_find_table_edge(req):
    print("Trying to find table edge...")
    filter_image(req.point_cloud)
    return

def callback_find_planes(req):
    print("Trying to find table plane...")
    return

def callback_recog_objs(req):
    print("Trying to recognize object by color")
    return
                    
def main():
    global bgr
    print("PRACTICE 05 - " + NAME)
    rospy.init_node("practice05")
    rospy.Service("/vision/line_finder/find_table_edge", FindLines, callback_find_table_edge);
    rospy.Service("/vision/line_finder/find_horizontal_plane_ransac", FindPlanes, callback_find_planes);
    rospy.Service("/vision/obj_reco/recognize_objects" , RecognizeObjects, callback_recog_objs);
    loop = rospy.Rate(10)
    bgr = numpy.zeros((480, 640, 3), numpy.uint8)
    while not rospy.is_shutdown() and cv2.waitKey(10) != 27:
        cv2.imshow("bgr", bgr)
        cv2.waitKey(1)
        loop.sleep()

if __name__ == '__main__':
    main()
