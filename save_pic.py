# coding:utf-8
import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cv2.namedWindow("Capture", 0)
cv2.moveWindow("Capture", 0, 0)
cv2.resizeWindow("Capture", 1920, 1080)

flag = cap.isOpened()
index = 0
count = 923
cap.set(3, 1920)
cap.set(4, 1080)
while (flag):
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)
    k = cv2.waitKey(1) & 0xFF
    if index % 3 == 0:
        cv2.imwrite("D:/05-project/01-code/Project Code/22-auto-oil/auto-fuel/runs/save/" + str(count) + ".jpg", frame)
        print("save" + str(count) + ".jpg successfully!")
        count += 1
        print("-------------------------")
    index += 1
cap.release()
cv2.destroyAllWindows()
