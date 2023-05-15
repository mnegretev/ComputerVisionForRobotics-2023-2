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

NAME = "SERGIO ALVARADO RAMOS"

def find_plane_by_ransac(points, min_points, tolerance, max_attempts):
    #
    # TODO:
    #
    # Find a plane given by a point and a normal vector. You can use the following steps:
    #
    # Set number_of_inliers to zero
    # WHILE number_of_inliers < min_points and max_attempts > 0:
    #    Get p1,p2,p3 as random samples of the set of points
    #    Calculate the plane center as the mean of the three points
    #    Calculate the normal to the three points (p1 - p2)x(p1 - p3)
    #    Calculate de distance to the candidate plane for each point p in points
    #    Get all points with a distance less than tolerance
    #    Get the number_of_inliers
    #    Decrease attempts by one
    # Get the set P of all points with distance to plane less than tolerance
    # Get eigenvalues and eigenvectors of the covariance matrix of P
    # Return the following values:
    # mean_point, normal_to_plane, number_of_inliers, min_point, max_point
    inliers_counting = 0
    mean = numpy.asarray([0,0,0])
    n=points.shape
    print(n)
    while(inliers_counting<min_points and max_attempts>0):
        inliers_counting=0
        p1,p2,p3=points[numpy.random.randint(n[0], size=3)] #points 1,2,3
        pc=(p1+p2+p3)/3 #plane center
        normal=numpy.cross( (p1-p2), (p1-p3) )
        normal=normal/numpy.linalg.norm(normal)
        d=numpy.zeros(n[0])
        for i in range(n[0]):
            d[i]=abs(numpy.dot(normal, points[i,:]-pc) )
                #print(d[i])
                #print(normal)
                #print(points[i,:])
            if d[i]< tolerance:
                inliers_counting+=1
        mask = (d[:] < tolerance)
        max_attempts-=1
    P=points[mask,:]
    print(P.shape)
    print(pc)
    A=numpy.cov(numpy.transpose(P))
    print(inliers_counting)
    #print(points[mask,:])
    mean=numpy.mean(P, axis=0)
    #mean = numpy.asarray([0,0,0])
    #normal = numpy.asarray([0,0,0])
    W,V=numpy.linalg.eig(A)
    print(W)
    print(V)
    #i=numpy.argmin(W)
    j=numpy.argmax(W)
    Qpca=V[:,j]
    Ppca=numpy.dot(P, Qpca)
    i_min=numpy.argmin(Ppca)
    i_max=numpy.argmax(Ppca)
    [x1,y1,z1]=P[i_min,:]
    [x2,y2,z2]=P[i_max,:]
    min_p=[x1,y1,z1]
    max_p=[x2,y2,z2]
    print(min_p)
    print(max_p)
    return mean, normal, inliers_counting, min_p, max_p

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
    mean, normal, n_points, min_point, max_point = find_plane_by_ransac(xyz, 18000, 0.03, 100)
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
