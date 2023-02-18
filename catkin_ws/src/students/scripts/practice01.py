#
# COMPUTER VISION FOR ROBOTICS - FI-UNAM - 2023-2
# PRACTICE 01 - THE OPENCV LIBRARY
#


import numpy as np
import cv2
from statistics import mean

def Trackbar(val):
    global circle_radius
    circle_radius = val

cap  = cv2.VideoCapture(0)
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #cv2.imshow('My Video', frame)
    
#############################################
    class BoundingBoxWidget(object):

        def __init__(self):
        
            global circle_radius
            circle_radius=10

            self.original_image = frame
           

            # self.original_image=cv2.resize(self.original_image,(800,687))
        
            self.playa = cv2.imread('universo.jpg')

            self.clone = self.original_image.copy()

            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.extract_coordinates)
       
            # Bounding box reference points
            self.image_coordinates = []
    


        def extract_coordinates(self, event, x, y, flags, parameters):
        
        # Record starting (x,y) coordinates on left mouse button click
            if event == cv2.EVENT_LBUTTONDOWN:
                global image_coordinates
                image_coordinates =np.array([x,y])
                print('cordenadas1 : ',self.image_coordinates)
                

        # Record ending (x,y) coordintes on left mouse button release
            if event == cv2.EVENT_LBUTTONUP:
                

                self.image_coordinates2=np.array([x,y])
                print('cordenadas2 : ',self.image_coordinates2)

                rs=np.append(image_coordinates,self.image_coordinates2)
                print("rs : ",rs)
           
            
                zx=rs[0]
                zy=rs[1]
                zw=rs[2]
                zh=rs[3]
                print(zx,zy,zw,zh)
           

            # Dibujando rectangulo
                #cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (255,0,0), 2) 
                #cv2.imshow("image", self.clone)

            #Dejando solo la zona marcada 
                Rectangulo=self.original_image[zy:zh,zx:zw]
                # cv2.imshow("Rectangulo",Rectangulo)

            ## Promedio
                p0=cv2.mean(Rectangulo)[0]
                p1=cv2.mean(Rectangulo)[1]
                p2=cv2.mean(Rectangulo)[2]

                print('Promedio 1: ', p0)
                print('Promedio 2: ', p1)
                print('Promedio 3: ', p2)

                global circle_radius
                cv2.createTrackbar('r','image',circle_radius, 100, Trackbar)
                ValorTrackbar= int(cv2.getTrackbarPos('r','image'))
                print("Trackbar : ",ValorTrackbar)
            

            ###umbral
                Umin1=p0-ValorTrackbar
                Umax1=p0+ValorTrackbar

                Umin2=p1-ValorTrackbar
                Umax2=p1+ValorTrackbar

                Umin3=p2-ValorTrackbar
                Umax3=p2+ValorTrackbar
            
                print('Umin1,Umax1, : ({}, {},)'.format(Umin1,Umax1,))
                print('Umin2,Umax2, : ({}, {},)'.format(Umin2,Umax2,))
                print('Umin3,Umax3, : ({}, {},)'.format(Umin3,Umax3,))


                rBajo = np.array([Umin1, Umin2, Umin3], np.uint8)
                rAlto = np.array([Umax1, Umax2, Umax3], np.uint8)
        

                BGR=cv2.inRange(self.original_image,rBajo, rAlto)
                BGR=BGR.astype(np.uint8)
          
                img_not    = cv2.bitwise_not(BGR)


                resize=cv2.resize(self.playa,(800,480))
           

                fondo=cv2.bitwise_and(resize,resize,mask=BGR)
            
                cv2.imshow('FONDO',fondo)
          
            
                invertida=cv2.bitwise_and(self.original_image,self.original_image,mask=img_not)
                cv2.imshow('Angel',invertida)
                global final

                final=cv2.add(fondo,invertida)
                cv2.imshow('Imagen Final',final)
                
                
        


               




        def show_image(self):
            return self.clone
  
        
# if __name__ == '__main__':

#      boundingbox_widget = BoundingBoxWidget()
     
#      while True:
         
#          cv2.imshow('image', boundingbox_widget.show_image())
#          key = cv2.waitKey(0)

################################################
    
    
    boundingbox_widget = BoundingBoxWidget()
    cv2.imshow('image', boundingbox_widget.show_image())

    if cv2.waitKey(10) & 0xFF == 27:
        break
    
    


cap.release(0)
cv2.destroyAllWindows()

