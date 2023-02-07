import cv2
import numpy as np
import pandas as pd
import os
import json
import sys
import pathlib


class cutPCB:

    def __init__(self, ):

        self.current_ngt_index = 0
        self.imageIndex = 0
        self.imageOnDisplay = None
        self.output_path = os.path.join(os.path.dirname(sys.argv[0]), "parts")
        self.categories = ["FalschPositiv","Typ1", "Typ2", "Typ3", "Typ4", "Typ5", "Typ6", "Typ7", "Typ8", "Typ9", "Typ10"]


        #Load Categories from config.json
        configpath= os.path.join(os.path.dirname(sys.argv[0]), "config.json")
        print(configpath)
        if os.path.exists(configpath):
            with open(configpath, 'r') as f:
                try:
                    self.categories = json.load(f)
                except:
                    print("Error loading config.json. Please check if it is valid JSON.")

        else:
            print("config.json not found. Using default categories.")
            with open(configpath, 'w') as f:
                json.dump(self.categories, f)

        #Create Output Folders
        for category in self.categories:
            os.makedirs(os.path.join(self.output_path, category), exist_ok=True)

        #Recursivly find all ngt files in the current directory
        input_path = pathlib.Path(sys.argv[0]).parent
        self.ngt_list = list(input_path.rglob("*.ngt"))
        if(len(self.ngt_list) == 0):
            print("No ngt files found. Please place ngt files in the same directory as this script.")
            sys.exit()

        current_ngt_index = 0

        self.color, self.df, self.imagename = self.load_data()


    def imcrop(self, img, bbox):
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return img[y1:y2, x1:x2, :].copy()

    def pad_img_to_fit_bbox(self, img, x1, x2, y1, y2):
        img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0), -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_CONSTANT, None, (255, 0, 0))
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
        return img, x1, x2, y1, y2

    def load_data(self, ):

        if self.current_ngt_index >= len(self.ngt_list):
            return np.array([]), pd.DataFrame(), "None"

        imagename = pathlib.Path(self.ngt_list[self.current_ngt_index]).stem
        image_dir = pathlib.Path(pathlib.Path(self.ngt_list[self.current_ngt_index]).parent.joinpath("..","checkimg")).resolve()

        R = cv2.imread(str(image_dir.joinpath(f"R_{imagename}.jpg")),cv2.IMREAD_GRAYSCALE)
        G = cv2.imread(str(image_dir.joinpath(f"G_{imagename}.jpg")),cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(str(image_dir.joinpath(f"B_{imagename}.jpg")),cv2.IMREAD_GRAYSCALE)

        #color = cv2.UMat(cv2.merge([B, G, R]))
        color = cv2.merge([B, G, R])

        col_names = ["index", "X1", "Y1", "Klasse", "Flaeche", "X2", "X3", "Y2", "Y3", "F1", "F2"]
        df=pd.read_csv(self.ngt_list[self.current_ngt_index], sep=',',index_col=0, header=None, names=col_names, skiprows=2)
        print(f"Loaded {imagename} ({self.current_ngt_index+1}/{len(self.ngt_list)})")

        self.current_ngt_index += 1
        return color, df, imagename

    def getNextImage(self, catIndex = -1):            


        if self.imageIndex >= len(self.df.index - 1):
            self.imageIndex = 0
            self.color, self.df, self.imagename = self.load_data()
            if self.color.size == 0:
                cv2.displayStatusBar('Display', "Keine weiteren Dateien mehr vorhanden. Bitte mit ESC beenden.")
                return
            
        cv2.displayStatusBar('Display', f"Aktuelle Datei: {self.imagename} - {self.imageIndex}/{len(self.df.index)-1} Gesamt: {self.current_ngt_index}/{len(self.ngt_list)}")
        
        if catIndex >= 0:
            path = os.path.join(self.output_path,self.categories[catIndex],f"{self.imagename}_{self.imageIndex}_{str(catIndex)}.jpg")
            cv2.imwrite(path, self.imageOnDisplay, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
        row = self.df.iloc[self.imageIndex]
        topleft = np.array([row['X1'], row['Y1']])
        topleft = np.floor_divide(topleft, 200) * 200
        topleft = topleft -200
        bottomright = topleft + 600

        self.imageOnDisplay = self.imcrop(self.color, (topleft[0], topleft[1], bottomright[0], bottomright[1]))
        output = self.imageOnDisplay.copy()
        cv2.rectangle(output, (row.X2-topleft[0], row.Y2-topleft[1]), (row.X3-topleft[0], row.Y3-topleft[1]), (255, 0, 255),1)
        cv2.putText(output, str(row.F1), (row.X1+10-topleft[0], row.Y1+10-topleft[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        self.imageIndex += 1

        return output


cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

cPCB = cutPCB()

def button_callback(*args):

    nextImage = cPCB.getNextImage(args[1])
    if nextImage is not None:
        cv2.imshow('Display', nextImage)

for index, category in enumerate(cPCB.categories):
    cv2.createButton(category,button_callback,index,cv2.QT_PUSH_BUTTON,1)


cv2.imshow('Display', cPCB.getNextImage())

while cv2.getWindowProperty('Display', cv2.WND_PROP_VISIBLE) > 0:

    key = cv2.waitKey(20) & 0xff
    if key==ord('q') or key == 27:
        break

cv2.destroyAllWindows()
