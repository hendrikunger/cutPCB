#%%
from PIL import Image, ImageDraw, ImageFont
import pandas as pd



#%%
Image.MAX_IMAGE_PIXELS = None
#imagename = "NG22_22"
imagename = "NG5_5"
#imagename = "NG11_11"
#imagename = "NG14_14"

extension = ".jpg"
#%%
red    = Image.open("R"+"_"+imagename+extension)
green  = Image.open("G"+"_"+imagename+extension)
blue   = Image.open("B"+"_"+imagename+extension)

rgb = Image.merge("RGB",(red,green,blue))
rgb.save("RGB.jpg", quality=100)
# %%
col_names = ["index", "X1", "Y1", "Klasse", "Flaeche", "X2", "X3", "Y2", "Y3", "F1", "F2"]
df=pd.read_csv(imagename+".ngt", sep=',',index_col=0, header=None, names=col_names, skiprows=2)

# %%
def point(canvas, x, y , size=50, color="red"):
    coordsa = zip(x-size, y-size)
    coordsb = zip(x+size, y+size)
    for a,b in zip(coordsa, coordsb):
        #print([a,b])
        canvas.rectangle([a,b], fill=color)

def rect(canvas, x1, y1, x2, y2 , color="red"):
    coordsa = zip(x1, y1)
    coordsb = zip(x2, y2)
    for a,b in zip(coordsa, coordsb):
        #print([a,b])
        canvas.rectangle([a,b], outline=color, width=3)

def annotate(canvas, x, y , text, color="red", size=30):
    font = font = ImageFont.truetype("arial.ttf", size=size)
    for a,b, _text in zip(x+10, y+10, text):
        canvas.text([a,b], str(_text), fill=color, font=font,  anchor="ms")

 # %%
copy_of_rgb = rgb.copy()
draw = ImageDraw.Draw(copy_of_rgb)
rect(draw, df.X2, df.Y2, df.X3, df.Y3, "magenta")
point(draw, df.X1, df.Y1, 2, "red")
point(draw, df.X2, df.Y2, 1, "blue")
point(draw, df.X3, df.Y3, 1, "green")
annotate(draw, df.X1, df.Y1, df.F1, "red", size=20)




copy_of_rgb.save("RGB_annotated.jpg", quality=80)
# %%
# %%
