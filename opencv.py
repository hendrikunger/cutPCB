#%%
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import imutils

#imagename = "NG22_22"
imagename = "NG5_5"
#imagename = "NG11_11"
#imagename = "NG14_14"

extension = ".jpg"
#%%
R = cv2.imread("R"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)
G = cv2.imread("G"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)
B = cv2.imread("B"+"_"+imagename+extension,cv2.IMREAD_GRAYSCALE)

color = cv2.merge([B, G, R])
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
threshInv, mask  = cv2.threshold(gray, 0, 255 ,cv2.THRESH_BINARY)

plt.axis("off")
#plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
plt.show()
#%%
tcontours, hierachry = cv2.findContours(mask, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
contours = []
for i,c in enumerate(tcontours):
  if len(c) > 2: contours.append(c)

#%%
print(len(contours))


output = color.copy()
#%%

for i, c in enumerate(contours):
  M = cv2.moments(c)
  #cX = int((M["m10"] / M["m00"]))
  #cY = int((M["m01"] / M["m00"]))
  drawcolor = random.sample(range(1, 256), 3)
  cv2.drawContours(output, [c], -1, drawcolor, 30)
  #cv2.putText(output, i, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,		0.5, drawcolor, 2)
#%%
plt.axis("off")
#plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
plt.imshow(output)
plt.show()

cv2.imwrite('Test_gray.jpg', output)
# %%
