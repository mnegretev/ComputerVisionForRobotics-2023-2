import cv2
import numpy

def main():
    img_blank  = numpy.zeros((512, 512, 3), numpy.uint8)
    img_blank[128:256,0:256] = 255*numpy.ones((128, 256, 3), numpy.uint8)
    img_baboon = cv2.imread('baboon.jpg')
    img_and    = cv2.bitwise_and(img_baboon, img_blank)
    img_or     = cv2.bitwise_or (img_baboon, img_blank)
    img_not    = cv2.bitwise_not(img_baboon)
    img_xor    = cv2.bitwise_xor(img_baboon, img_blank)
    cv2.imshow("OR Operator", img_or)
    cv2.imshow("XOR Operator", img_xor)
    cv2.imshow("AND Operator", img_and)
    cv2.imshow("NOT Operator", img_not)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

