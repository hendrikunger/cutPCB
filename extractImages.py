#%%
import cv2
import numpy as np
import random
import pandas as pd
import os


def imcrop(img, bbox):
   x1, y1, x2, y2 = bbox
   if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
   return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
   img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0), -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_CONSTANT, None, (255, 0, 0))
   y2 += -min(0, y1)
   y1 += -min(0, y1)
   x2 += -min(0, x1)
   x1 += -min(0, x1)
   return img, x1, x2, y1, y2
path = os.path.join(os.path.dirname(__file__), "parts")
os.makedirs(path, exist_ok=True) 

#imagename = "NG22_22"
imagename = "NG5_5"
#imagename = "NG11_11"
#imagename = "NG14_14"

ANNOTATE = True

extension = ".jpg"
#%%
R = cv2.imread("R"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)
G = cv2.imread("G"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)
B = cv2.imread("B"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)

#color = cv2.UMat(cv2.merge([B, G, R]))
color = cv2.merge([B, G, R])

# %%
col_names = ["index", "X1", "Y1", "Lage im PCB", "Flaeche", "X2", "X3", "Y2", "Y3", "F1", "F2"]
df=pd.read_csv(imagename+".ngt", sep=',',index_col=0, header=None, names=col_names, skiprows=2)

# %%

annotated = color.copy()

for row in df.itertuples():
   topleft = np.array([row.X1, row.Y1])
   topleft = np.floor_divide(topleft, 200) * 200
   topleft = topleft -200
   bottomright = topleft + 600
   drawcolor = random.sample(range(1, 256), 3) 
   #cv2.rectangle(annotated, topleft, bottomright, drawcolor,5)
   if ANNOTATE:
      cv2.rectangle(annotated, (row.X2, row.Y2), (row.X3, row.Y3), (255, 0, 255),1)
      cv2.putText(annotated, str(row.F1), (row.X1+10, row.Y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      #cv2.circle(annotated,(row.X1,row.Y1), 10, (0,0,255), 20)

   output = imcrop(annotated, (topleft[0], topleft[1], bottomright[0], bottomright[1]))
   cv2.imwrite(os.path.join(path,f"{row.Index}.jpg"), output, [cv2.IMWRITE_JPEG_QUALITY, 100])
   #print(os.path.join(path,f"{row.Index}.jpg"))


#cv2.imwrite(path + '/Full.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 100])





