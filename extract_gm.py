#%%import cv2
import numpy as np
import random
import pandas as pd
import os
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as compare_ssim

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



#%%
rootdir = os.path.dirname(__file__)
gmdir= os.path.join(rootdir,"gm")


for f in os.listdir(gmdir):
    if os.path.isdir(f):
        print(f)

























base_path = os.path.join(rootdir,"614581_helles_Entek_neu")


print(base_path)

img_path=os.path.join(base_path,"614581","3031341_1_A_1","checkimg")
ngt_path=os.path.join(base_path,"614581","3031341_1_A_1","ngpointdata")
col_names = ["index", "X1", "Y1", "Klasse", "Flaeche", "X2", "X3", "Y2", "Y3", "F1", "F2"]

ANNOTATE=True

#%%

path = os.path.join(base_path,"curdata_st4","camera1")

R = cv2.imread(os.path.join(path,"KIBANCUR_R.jpg"),cv2.IMREAD_GRAYSCALE)
G = cv2.imread(os.path.join(path,"KIBANCUR_G.jpg"),cv2.IMREAD_GRAYSCALE)
B = cv2.imread(os.path.join(path,"KIBANCUR_B.jpg"),cv2.IMREAD_GRAYSCALE)

master_1 = cv2.merge([B, G, R])

#gm_top = cv2.imread(os.path.join(base_path,"curdata_st4","combined_top.jpg"),cv2.IMREAD_COLOR)
#gm_bottom = cv2.imread(os.path.join(base_path,"curdata_st4","combined_bottom.jpg"),cv2.IMREAD_COLOR)


#%%
ngt_list = [x.stem for x in Path(ngt_path).rglob("*.ngt")]


#%%

for file in ngt_list[2:3]:
    print(file)
    df=pd.read_csv(os.path.join(ngt_path, file+".ngt"), sep=',',index_col=0, header=None, names=col_names, skiprows=2)

    R = cv2.imread(os.path.join(img_path,"R"+"_"+file+".jpg"),cv2.IMREAD_GRAYSCALE)
    G = cv2.imread(os.path.join(img_path,"G"+"_"+file+".jpg"),cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(os.path.join(img_path,"B"+"_"+file+".jpg"),cv2.IMREAD_GRAYSCALE)

    color = cv2.merge([B, G, R])
    annotated = color.copy()

    for row in df.itertuples():
        topleft = np.array([row.X1, row.Y1])

        topleft = np.floor_divide(topleft, 200) * 200
        topleft = topleft -200
        bottomright = topleft + 600



        error_img = imcrop(color, (topleft[0], topleft[1], bottomright[0], bottomright[1]))
        error_img = cv2.GaussianBlur(error_img, (3,3), 0)
        
        margin = 0
        gm_img = imcrop(master_1, (topleft[0] - margin, topleft[1] - margin, bottomright[0] + margin, bottomright[1] + margin))
        gm_img = cv2.GaussianBlur(gm_img, (5,5), 0)

        error_img_grey = cv2.cvtColor(error_img, cv2.COLOR_BGR2GRAY)
        gm_img_grey = cv2.cvtColor(gm_img, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(error_img_grey, gm_img_grey, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        blur = cv2.GaussianBlur(diff, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 23)

        absdiff = cv2.absdiff(gm_img_grey, error_img_grey)
        absdiff = cv2.GaussianBlur(absdiff, (3,3), 0)
        absdiffthresh = cv2.threshold(absdiff, 0, 255, cv2.THRESH_OTSU)[1]

        
        cv2.imwrite(os.path.join(rootdir,"result",f"merged_{file}_{row.Index}_m.jpg"), gm_img, [cv2.IMWRITE_JPEG_QUALITY, 95]) 
        cv2.imwrite(os.path.join(rootdir,"result",f"merged_{file}_{row.Index}_y1.jpg"), thresh, [cv2.IMWRITE_JPEG_QUALITY, 95]) 
        #cv2.imwrite(os.path.join(rootdir,"result",f"merged_{file}_{row.Index}_x.jpg"), absdiff, [cv2.IMWRITE_JPEG_QUALITY, 95]) 
        #cv2.imwrite(os.path.join(rootdir,"result",f"merged_{file}_{row.Index}_z.jpg"), absdiffthresh, [cv2.IMWRITE_JPEG_QUALITY, 95]) 
        cv2.imwrite(os.path.join(rootdir,"result",f"merged_{file}_{row.Index}_z1.jpg"), cv2.bitwise_and(error_img, error_img, mask=thresh), [cv2.IMWRITE_JPEG_QUALITY, 95])
        annotated = color.copy()
        if ANNOTATE:
            cv2.rectangle(annotated, (row.X2, row.Y2), (row.X3, row.Y3), (255, 0, 255),1)
            cv2.putText(annotated, str(row.F1), (row.X1+10, row.Y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            #cv2.circle(annotated,(row.X1,row.Y1), 10, (0,0,255), 20)
        error_img=imcrop(annotated, (topleft[0], topleft[1], bottomright[0], bottomright[1]))
        cv2.imwrite(os.path.join(rootdir,"result",f"merged_{file}_{row.Index}.jpg"), error_img, [cv2.IMWRITE_JPEG_QUALITY, 95]) 



# %%
