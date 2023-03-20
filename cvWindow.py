import cv2
import numpy as np
import pandas as pd
import os
import json
import sys
from pathlib import Path
from datetime import datetime


class cutPCB:

    def __init__(self, ):
        self.current_ngt_index = 0
        self.imageIndex = 0
        self.imageOnDisplay = None
        self.output_path = os.path.join(os.path.dirname(sys.argv[0]), "parts")
        self.categories = ["FalschPositiv","Typ1", "Typ2", "Typ3", "Typ4", "Typ5", "Typ6", "Typ7", "Typ8", "Typ9", "Typ10"]
        self.coords = {}
        self.state = {"path":"", "count":0, "current_ngt_index":0, "imageIndex":0}
        self.highlightErrors = False

        #Load Categories from config.json
        self.configpath= os.path.join(os.path.dirname(sys.argv[0]), "config.json")
        print(self.configpath)
        if os.path.exists(self.configpath):
            with open(self.configpath, 'r') as f:
                try:
                    file = json.load(f)
                    self.categories = file["categories"]
                    self.state = file["state"]
                except:
                    print("Error loading config.json. Please check if it is valid JSON.")

        else:
            print("config.json not found. Using default categories.")
            with open(self.configpath, 'w') as f:
                out = {"categories": self.categories, "state": {"path":"", "count":0, "current_ngt_index":0, "imageIndex":0}}
                json.dump(out, f)

        #Create Output Folders
        for category in self.categories:
            os.makedirs(os.path.join(self.output_path, category), exist_ok=True)

        #Recursivly find all ngt files in the current directory
        self.input_path = Path(sys.argv[0]).parent
        self.ngt_list = list(self.input_path.rglob("*.ngt"))
        if(len(self.ngt_list) == 0):
            print("No ngt files found. Please place ngt files in the same directory as this script.")
            sys.exit()

        if self.state["path"] == str(self.input_path) and self.state["count"] ==  len(self.ngt_list):
            self.current_ngt_index = self.state["current_ngt_index"]
            self.imageIndex = self.state["imageIndex"]
            print(f"Resuming from last session., Current NGT index: {self.current_ngt_index}, Image: {self.imageIndex}")
        else:
            print("No saved Session or missmatching NGT file count. Starting new session.")
            self.current_ngt_index = 0

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

        imagename = Path(self.ngt_list[self.current_ngt_index]).stem
        image_dir = Path(Path(self.ngt_list[self.current_ngt_index]).parent.joinpath("..","checkimg")).resolve()

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

        if catIndex >= 0 and self.imageIndex <= len(self.df.index):
            filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}___{self.imagename}___{self.imageIndex}-{str(catIndex)}"
            path = os.path.join(self.output_path,self.categories[catIndex],filename)
            cv2.imwrite(path+".jpg", self.imageOnDisplay, [cv2.IMWRITE_JPEG_QUALITY, 100])
            #write error data to json
            with open(path+".json", "w") as outfile:
                json.dump(self.coords, outfile)

        if self.imageIndex >= len(self.df.index):
            self.imageIndex = 0
            self.color, self.df, self.imagename = self.load_data()
            if self.color.size == 0:
                cv2.displayStatusBar('Display', "Keine weiteren Dateien mehr vorhanden. Bitte mit ESC beenden.")
                self.imageIndex = np.inf
                return
    
        row = self.df.iloc[self.imageIndex]
        errorCoords = np.array([row['X1'], row['Y1']])
        topleft = np.floor_divide(errorCoords, 200) * 200
        topleft = topleft -200
        bottomright = topleft + 600

        self.imageOnDisplay = self.imcrop(self.color, (topleft[0], topleft[1], bottomright[0], bottomright[1]))

        self.coords["errorcoords"] = [int(row.X1-topleft[0]), int(row.Y1-topleft[1])]
        self.coords["topleft"] = [int(row.X2-topleft[0]), int(row.Y2-topleft[1])]
        self.coords["bottomright"] = [int(row.X3-topleft[0]), int(row.Y3-topleft[1])]
        self.coords["errortype"] = str(row.F1)

        if self.highlightErrors:
            output = self.highlightCurrentImage()
        else:
            output = self.imageOnDisplay

        with open(self.configpath, 'w') as f:
                out = {"categories": self.categories, "state": {"path":str(self.input_path), "count":len(self.ngt_list), "current_ngt_index": self.current_ngt_index-1, "imageIndex":self.imageIndex}}
                json.dump(out, f)

        self.imageIndex += 1
        cv2.displayStatusBar('Display', f"Aktuelle Datei: {self.imagename} - {self.imageIndex}/{len(self.df.index)} Gesamt: {self.current_ngt_index}/{len(self.ngt_list)}")

        return output 
    
    def highlightCurrentImage(self):
        output = self.imageOnDisplay.copy()
        cv2.rectangle(output, self.coords["topleft"], self.coords["bottomright"], (255, 0, 255),1)
        cv2.putText(output, self.coords["errortype"], (self.coords["errorcoords"][0]+10, self.coords["errorcoords"][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return output
    

cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
cPCB = cutPCB()

def button_callback(*args):
    nextImage = cPCB.getNextImage(args[1])
    if nextImage is not None:
        cv2.setTrackbarPos("NGT Datei Nummer", '', cPCB.current_ngt_index)
        cv2.imshow('Display', nextImage)


def button_load_callback(*args):
    cPCB.current_ngt_index = cv2.getTrackbarPos("NGT Datei Nummer", '') - 1
    cPCB.color, cPCB.df, cPCB.imagename = cPCB.load_data()
    cPCB.imageIndex = 0
    cv2.imshow('Display', cPCB.getNextImage())

def button_highlight_callback(state, *args):   
    cPCB.highlightErrors = state
    if state:
        cv2.imshow('Display', cPCB.highlightCurrentImage())
    else:
        cv2.imshow('Display', cPCB.imageOnDisplay)

def trackbar_callback(*args):
    pass


cv2.createButton("Fehler markieren",button_highlight_callback,0,cv2.QT_CHECKBOX,0)
cv2.createButton("NGT Datei von Slider laden",button_load_callback,0,cv2.QT_PUSH_BUTTON,1)
cv2.createTrackbar("NGT Datei Nummer", '', 1, len(cPCB.ngt_list), trackbar_callback)
cv2.setTrackbarMin("NGT Datei Nummer", '', 1)

for index, category in enumerate(cPCB.categories):
    cv2.createButton(category,button_callback,index,cv2.QT_PUSH_BUTTON,1)




cv2.imshow('Display', cPCB.getNextImage())

while cv2.getWindowProperty('Display', cv2.WND_PROP_VISIBLE) > 0:

    key = cv2.waitKey(20) & 0xff
    if key==ord('q') or key == 27:
        break

cv2.destroyAllWindows()
