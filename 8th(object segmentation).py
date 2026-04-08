import cv2

image = cv2.imread("download.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

cv2.imshow("Segmented", output)
cv2.waitKey(0)
cv2.destroyAllWindows()