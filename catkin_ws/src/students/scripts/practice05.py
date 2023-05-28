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
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, Vector3, Quaternion
from vision_msgs.srv import FindLines, FindLinesResponse
from vision_msgs.srv import FindPlanes, FindPlanesResponse
from vision_msgs.srv import RecognizeObjects
from visualization_msgs.msg import Marker

NAME = "FULL_NAME"

def find_plane_by_ransac(points, min_points, tolerance, max_attempts):
    
    number_of_inliers = 0

    while number_of_inliers < min_points and max_attempts > 0:

        random_indices = numpy.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[random_indices]

        center = (p1 + p2 + p3) / 3

        normal = numpy.cross(p1 - p2, p1 - p3)

        distances = numpy.abs(numpy.dot(points - center, normal))

        inliers = points[distances < tolerance]

        number_of_inliers = len(inliers)

        max_attempts -= 1

    P = points[distances < tolerance]

    covariance_matrix = numpy.cov(P, rowvar=False)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)

    mean_point = numpy.mean(P, axis=0)
    normal_to_plane = eigenvectors[:, numpy.argmin(eigenvalues)]
    min_point = numpy.min(P, axis=0)
    max_point = numpy.max(P, axis=0)

    return mean_point, normal_to_plane, number_of_inliers, min_point, max_point

def get_plane_marker(min_p, max_p):
    marker = Marker();
    marker.header.frame_id = "base_link";
    marker.header.stamp = rospy.Time.now();
    marker.ns = "obj_reco_markers";
    marker.id = 1;
    marker.type = Marker.CUBE;
    marker.action = Marker.ADD;
    marker.pose.position = Point(x=(min_p[0]+max_p[0])/2, y=(min_p[1]+max_p[1])/2, z=(min_p[2]+max_p[2])/2)
    marker.pose.orientation = Quaternion(x=0,y=0,z=0,w=1.0)
    marker.scale = Vector3(x=(max_p[0]-min_p[0]), y=(max_p[1]-min_p[1]), z=(max_p[2]-min_p[2]))
    marker.color = ColorRGBA(a=0.5, r=0.0, g=0.5, b=0.0)
    marker.lifetime = rospy.Duration(10.0);
    return marker
    
def callback_find_planes(req):
    global pub_marker
    print("Trying to find table plane...")
    xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(req.point_cloud)
    xyz = xyz[(xyz[:,0] > 0.3) & (xyz[:,0] < 2.0) & (xyz[:,1] > -2.0) & (xyz[:,1] < 2.0) & (xyz[:,2] > 0.5) & (xyz[:,2] < 1.5)]
    mean, normal, n_points, min_point, max_point = find_plane_by_ransac(xyz, 150000, 0.03, 5)
    pub_marker.publish(get_plane_marker(min_point, max_point))
    print("Found plane with mean " + str(mean) + " and normal " + str(normal))
    return FindPlanesResponse()

def callback_recog_objs(req):
    print("Trying to recognize object by color")
    return
                    
def main():
    global pub_marker
    print("PRACTICE 05 - " + NAME)
    rospy.init_node("practice05")
    rospy.Service("/vision/line_finder/find_horizontal_plane_ransac", FindPlanes, callback_find_planes);
    rospy.Service("/vision/obj_reco/recognize_objects" , RecognizeObjects, callback_recog_objs);
    pub_marker = rospy.Publisher("/vision/obj_reco/markers", Marker, queue_size=1);
    loop = rospy.Rate(10)
    while not rospy.is_shutdown():
        loop.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
