#%%
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd

#imagename = "NG22_22"
imagename = "NG5_5"
#imagename = "NG11_11"
#imagename = "NG14_14"

extension = ".jpg"
#%%
R = cv2.imread("R"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)
G = cv2.imread("G"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)
B = cv2.imread("B"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)

#color = cv2.UMat(cv2.merge([B, G, R]))
color = cv2.merge([B, G, R])
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

threshInv, mask  = cv2.threshold(gray, 1, 255 ,cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#cv2.imwrite('RGB_gray.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 100])
#cv2.imwrite('RGB_mask.jpg', mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
# %%
col_names = ["index", "X1", "Y1", "Klasse", "Flaeche", "X2", "X3", "Y2", "Y3", "F1", "F2"]
df=pd.read_csv(imagename+".ngt", sep=',',index_col=0, header=None, names=col_names, skiprows=2)

# %%
#plt.axis("off")
#plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
#plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
#plt.show()
#%%
tcontours, hierachry = cv2.findContours(mask, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
contours = []
for i,c in enumerate(tcontours):
  
  if (len(c) > 3) and (cv2.contourArea(c)>20): contours.append(c)

#%%
print(len(contours))



#%%
output = color.copy()
for i, c in enumerate(contours):
  M = cv2.moments(c)
  cX = int((M["m10"] / M["m00"]))
  cY = int((M["m01"] / M["m00"]))
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.04 * peri, True)
  drawcolor = random.sample(range(1, 256), 3)   
  cv2.drawContours(output, [c], -1, drawcolor, 50)
  print(len(c))
  #cv2.putText(output, i, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,		0.5, drawcolor, 2)
#%%
#plt.axis("off")
#plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
#plt.imshow(output)
#plt.show()
cv2.imwrite('RGB.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.imwrite('RGB_Bounds.jpg', output)
# %%
annotated = color.copy()
for row in df.itertuples():
  cv2.rectangle(annotated, (row.X2, row.Y2), (row.X3, row.Y3), (255, 0, 255),1)
  cv2.circle(annotated,(row.X1,row.Y1), 10, (0,0,255), -1)
  cv2.circle(annotated,(row.X2,row.Y2), 5, (255,0,0), -1)
  cv2.circle(annotated,(row.X3,row.Y3), 5, (0,255,0), -1)
  cv2.putText(annotated, str(row.F1), (row.X1, row.Y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
  # annotate(draw, row.X1, row.Y1, row.F1, "red", size=20)

cv2.imwrite('RGB_Annotated.jpg', annotated)
# %%
annotated = color.copy()
for row in df.itertuples():
  topleft = np.array([row.X1, row.Y1])
  topleft = np.floor_divide(topleft, 200) * 200
  topleft = topleft -200
  bottomright = topleft + 600
  drawcolor = random.sample(range(1, 256), 3) 
  cv2.rectangle(annotated, topleft, bottomright, drawcolor,5)
  cv2.circle(annotated,(row.X1,row.Y1), 10, drawcolor, 20)
  #cv2.putText(annotated, str(row.F1), (row.X1, row.Y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
  # annotate(draw, row.X1, row.Y1, row.F1, "red", size=20)

cv2.imwrite('RGB_Rounded.jpg', annotated)
# %%
