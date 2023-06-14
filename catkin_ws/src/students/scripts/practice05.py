#!/usr/bin/env python3
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRÁCTICA 05 - SEGMENTACIÓN DE LÍNEAS Y PLANOS MEDIANTE RANSAC Y PCA
#
import numpy as np
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

NAME = "TU_NOMBRE_COMPLETO"

def find_plane_by_ransac(points, min_points, tolerance, max_attempts):
    number_of_inliers = 0
    print("1")
    while number_of_inliers < min_points and max_attempts > 0:
        # Obtener p1, p2, p3 como muestras aleatorias del conjunto de puntos
        print("2")
        indices = np.random.choice(len(points), size=3, replace=False)
        p1, p2, p3 = points[indices[0]], points[indices[1]], points[indices[2]]
        
        # Calcular el centro del plano como el promedio de los tres puntos
        centro = (p1 + p2 + p3) / 3
        
        # Calcular el vector normal a partir de los tres puntos (p1 - p2) x (p1 - p3)
        normal = np.cross(p1 - p2, p1 - p3)
        normal /= np.linalg.norm(normal)
        print("3")
        # Calcular la distancia al plano candidato para cada punto p en points
        distancias = np.abs(np.dot(points - centro, normal))
        
        # Obtener todos los puntos con una distancia menor que la tolerancia
        inliers = points[distancias < tolerance]
        number_of_inliers = len(inliers)
        
        max_attempts -= 1
    print("4")
    # Obtener el conjunto P de todos los puntos con distancia al plano menor que la tolerancia
    P = points[distancias < tolerance]
    print("5")
    # Obtener los valores y vectores propios de la matriz de covarianza de P
    _, _, V = np.linalg.svd(P - np.mean(P, axis=0))
    print("6")
    # Devolver los siguientes valores:
    mean_point = np.mean(P, axis=0)
    normal_to_plane = V[-1]  # El último vector propio corresponde a la normal
    number_of_inliers = len(P)
    print("7")
    min_point = np.min(P, axis=0)
    max_point = np.max(P, axis=0)
    
    return mean_point, normal_to_plane, number_of_inliers, min_point, max_point 

def get_plane_marker(min_p, max_p):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "obj_reco_markers"
    marker.id = 1
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position = Point(x=(min_p[0] + max_p[0]) / 2, y=(min_p[1] + max_p[1]) / 2, z=(min_p[2] + max_p[2]) / 2)
    marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1.0)
    marker.scale = Vector3(x=(max_p[0] - min_p[0]), y=(max_p[1] - min_p[1]), z=(max_p[2] - min_p[2]))
    marker.color = ColorRGBA(a=0.5, r=0.0, g=0.5, b=0.0)
    marker.lifetime = rospy.Duration(10.0)
    return marker
    
def callback_find_planes(req):
    global pub_marker
    print("Intentando encontrar el plano de la mesa...")
    xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(req.point_cloud)
    xyz = xyz[(xyz[:,0] > 0.3) & (xyz[:,0] < 2.0) & (xyz[:,1] > -2.0) & (xyz[:,1] < 2.0) & (xyz[:,2] > 0.5) & (xyz[:,2] < 1.5)]
    mean, normal, n_points, min_point, max_point = find_plane_by_ransac(xyz, 150000, 0.03, 5)
    pub_marker.publish(get_plane_marker(min_point, max_point))
    print("Plano encontrado con media " + str(mean) + " y normal " + str(normal))
    return FindPlanesResponse()

def callback_recog_objs(req):
    print("Intentando reconocer objetos por color")
    return
                    
def main():
    global pub_marker
    print("PRÁCTICA 05 - " + NAME)
    rospy.init_node("practice05")
    rospy.Service("/vision/line_finder/find_horizontal_plane_ransac", FindPlanes, callback_find_planes)
    rospy.Service("/vision/obj_reco/recognize_objects" , RecognizeObjects, callback_recog_objs)
    pub_marker = rospy.Publisher("/vision/obj_reco/markers", Marker, queue_size=1)
    loop = rospy.Rate(10)
    while not rospy.is_shutdown():
        loop.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
