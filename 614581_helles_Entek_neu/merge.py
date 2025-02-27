import cv2
import os
rootdir = os.path.dirname(__file__)

items = os.listdir(rootdir)

cameraFolders = ["camera1", "camera2", "camera3", "camera4"]

for folder in items:
    if os.path.isdir(folder):
        for camera in cameraFolders:
            print(os.path.join(rootdir,folder,camera))


            R = cv2.imread(os.path.join(rootdir,folder,camera,"KIBANCUR_R.jpg"),cv2.IMREAD_GRAYSCALE)
            G = cv2.imread(os.path.join(rootdir,folder,camera,"KIBANCUR_G.jpg"),cv2.IMREAD_GRAYSCALE)
            B = cv2.imread(os.path.join(rootdir,folder,camera,"KIBANCUR_B.jpg"),cv2.IMREAD_GRAYSCALE)

            #color = cv2.UMat(cv2.merge([B, G, R]))
            color = cv2.merge([B, G, R])
            cv2.imwrite(os.path.join(rootdir,folder,camera,f"merged_{folder}_{camera}.jpg"), color) 

