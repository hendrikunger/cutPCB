#%%import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
import json
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager

rootdir = os.path.dirname(__file__)
gm_dict = {}
ngt_folders = ["205018_helles Entek","205047_helles Entek","205048_dunkles Rogers", "614581_helles Entek"]
gmdir= os.path.join(rootdir,"gm")

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



def loadMaster():
    print("Loading Golden Master Files")

    for f in tqdm(os.listdir(gmdir)):
        projectPath = os.path.join(gmdir,f)
        if os.path.isdir(projectPath):
            projectNR = f.split("_")[0]
            gm_dict[projectNR] = {}
            for cameraFolder in tqdm(os.listdir(projectPath), leave=False):
                cameraPath = os.path.join(projectPath,cameraFolder)
                R = cv2.imread(os.path.join(cameraPath,"KIBANCUR_R.jpg"),cv2.IMREAD_GRAYSCALE)
                G = cv2.imread(os.path.join(cameraPath,"KIBANCUR_G.jpg"),cv2.IMREAD_GRAYSCALE)
                B = cv2.imread(os.path.join(cameraPath,"KIBANCUR_B.jpg"),cv2.IMREAD_GRAYSCALE)

                master = cv2.merge([B, G, R])
                #save golden master image
                gm_dict[projectNR][cameraFolder] = master





def processfile(f):
    inputPath = os.path.join(rootdir,f)
    print("Processing Folder: ", f)
    files = [x for x in os.listdir(inputPath) if x.endswith(".jpg") and not x.endswith("_master.jpg") and not x.endswith("_mask.jpg") and not x.endswith("_maskedImage.jpg") and not x.endswith("_match.jpg")]
    for file in tqdm(files, leave=False):
        
        #load image
        color = cv2.imread(os.path.join(inputPath,file),cv2.IMREAD_COLOR)

        #get relevant Numbers
        fsplit = file.split("_")
        ngtName = file.split("___")[1]
        ngtNumber = int(file.split("___")[2].split("-")[0])-1
        errorNr = fsplit[7]
        gmNR = fsplit[8]
        batchNR = fsplit[9].split(".")[0]

        dataFolder = None
        #match folder name in ngt_folders with gmNr
        for folder in ngt_folders:
            if folder.split("_")[0] == gmNR:
                dataFolder = folder
                break

        if dataFolder:
            dataFolderPath=Path(os.path.join(rootdir,dataFolder))
            ngt_list = list(dataFolderPath.rglob("*.ngt"))

            ngt_list = [x for x in ngt_list if x.stem == ngtName]

            bestMatch = None
            found = False

            #iterate over all ngt files in folder
            for ngt_file in ngt_list:
                df=pd.read_csv(ngt_file, sep=',',index_col=0, header=None, names=["index", "X1", "Y1", "Klasse", "Flaeche", "X2", "X3", "Y2", "Y3", "F1", "F2"], skiprows=2)

                if ngtNumber >= len(df):continue
                row = df.iloc[ngtNumber]
                imagePath = Path(ngt_file.parent.joinpath("..","checkimg")).resolve()

                #get corresponding image
                R = cv2.imread(os.path.join(imagePath, "R"+"_"+ngt_file.stem+".jpg"),cv2.IMREAD_GRAYSCALE)
                G = cv2.imread(os.path.join(imagePath, "G"+"_"+ngt_file.stem+".jpg"),cv2.IMREAD_GRAYSCALE)
                B = cv2.imread(os.path.join(imagePath, "B"+"_"+ngt_file.stem+".jpg"),cv2.IMREAD_GRAYSCALE)
                #merge
                candidate_color = cv2.merge([B, G, R])
                errorCoords = np.array([row['X1'], row['Y1']])
                topleft = np.floor_divide(errorCoords, 200) * 200
                topleft = topleft -200
                bottomright = topleft + 600

                candidate_color = imcrop(candidate_color, (topleft[0], topleft[1], bottomright[0], bottomright[1]))

                grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                candidate_grey = cv2.cvtColor(candidate_color, cv2.COLOR_BGR2GRAY)
                #compare images
                score = compare_ssim(grey, candidate_grey)

                if bestMatch is None or score > bestMatch:
                    bestMatch = score

                if score > 0.98:
                    #save best match
                    #cv2.imwrite(os.path.join(inputPath,f"{file.split('.')[0]}_match.jpg"), candidate_color, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    #find camera of best match
                    cameraString = imagePath.parts[-2][-3:]
                    
                    match cameraString:
                        case "A_1": camera = "camera1"
                        case "A_2": camera = "camera2"
                        case "B_1": camera = "camera3"
                        case "B_2": camera = "camera4"
                        case _: 
                            print(f"No Camera Found for {imagePath}")
                            break
                    
                    #get master coresponding par of golden master
                    gm_img = imcrop(gm_dict[gmNR][camera], (topleft[0], topleft[1], bottomright[0], bottomright[1]))
                    cv2.imwrite(os.path.join(inputPath,f"{file.split('.')[0]}_master.jpg"), gm_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    gm_img = cv2.GaussianBlur(gm_img, (5,5), 0)
                    gm_img_grey = cv2.cvtColor(gm_img, cv2.COLOR_BGR2GRAY)

                    grey = cv2.GaussianBlur(grey, (3,3), 0)

                    (score, diff) = compare_ssim(grey, gm_img_grey, full=True)

                    diff = (diff * 255).astype("uint8")
                    
                    blur = cv2.GaussianBlur(diff, (3,3), 0)
                    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    cv2.imwrite(os.path.join(inputPath,f"{file.split('.')[0]}_mask.jpg"), thresh, [cv2.IMWRITE_JPEG_QUALITY, 100]) 
                    cv2.imwrite(os.path.join(inputPath,f"{file.split('.')[0]}_maskedImage.jpg"), cv2.bitwise_and(color, color, mask=thresh), [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
                    #weite json file with new data
                    with open(os.path.join(inputPath,file.replace("jpg", "json")), "w") as fjson:
                        errorCoords_list = [int(row.X2-topleft[0]), int(row.Y2-topleft[1])]
                        topleft_local = [int(row.X2-topleft[0]), int(row.Y2-topleft[1])]
                        bottomright_local = [int(row.X3-topleft[0]), int(row.Y3-topleft[1])]
                        jsonData = {"errorcoords": errorCoords_list, "topleft": topleft_local, "bottomright": bottomright_local, "errortype": f.split("_")[0]}
                        jsonData["masterdata"] = {"topleft": topleft.tolist(), "bottomright": bottomright.tolist(), "camera":camera, "path": str(ngt_file), "errorNumber": ngtNumber}
                        json.dump(jsonData, fjson)
                        
                    found = True
                    break
            
            if not found:
                #write placeholder file
                print("No Match Found" )
                with open(os.path.join(inputPath,f"{file.split('.')[0]}_error.txt"), "w") as f:
                    f.write(f"No Match Found. SSIM: {bestMatch} \n {ngt_file}")




#%%    
loadMaster()
if __name__ == '__main__':


    
    
    print("Loading NGT Files")
    folders = [x for x in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir,x)) and x not in ngt_folders and not x == "gm"]
    process_map(processfile, folders, max_workers=2)


