import cv2

img = cv2.imread('yesterday_test.jpg')
img = cv2.resize(img,(640,480))

cv2.imshow('original', img)

# Averaging
# You can change the kernel size as you want
blurImg = cv2.blur(img, (15,15))
cv2.imshow('Averaging blur', blurImg)
# cv2.imwrite('blurred_img.jpg',blurImg)

#Gaussian Blurring
gblur= cv2.GaussianBlur(img,(15,15),0)
cv2.imshow('Gaussian Blur', gblur)

#Median Filtering
median = cv2.medianBlur(img,7)
cv2.imshow('Median Filtering', median)

#Bilateral filtering
bf = cv2.bilateralFilter(img,9,75,75)
cv2.imshow('BiLateral Filtering', bf)

cv2.waitKey(0)
cv2.destroyAllWindows()