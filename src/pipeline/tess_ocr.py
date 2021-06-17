import cv2 
import pytesseract
from pytesseract import Output
import numpy as np
import pandas as pd 

"""
Note: This is an example for how to create a dataframe with bounding boxes for a document. It is for demonstration purposes. 
For this project, 'ready-made' bounding interim2 is used that is found in 'data/raw/interim2'.

"""
image = "../../data/raw/img/339.jpg"
output = "../../figures/tess_339.jpg"
img = cv2.imread(image)

custom_config = r'--oem 3 --psm 6'

d = pytesseract.image_to_data(img, output_type=Output.DICT) #, config=custom_config)
print(d.keys())

"""
left is the distance from the upper-left corner of the bounding interim2, to the left border of the image.
top is the distance from the upper-left corner of the bounding interim2, to the top border of the image.
width and height are the width and height of the bounding interim2.
conf is the model's confidence for the prediction for the word within that bounding interim2.
If conf is -1, that means that the corresponding bounding interim2 contains a block of text, rather than just a single word.

"""
#for graph modeling
xmin,ymin,xmax,ymax,Object = [],[],[],[],[]
df = pd.DataFrame() 

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) >= 0.74:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #print(x,y,w,h)
        text = d['text'][i]
      
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        img = cv2.putText(img, text, (x, y - 1),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)


        img_height, img_width =  img.shape[0], img.shape[1]
         
        xmin.append(x)
        ymin.append(y)
        xmax.append(x + w)
        ymax.append(y + h) 
        Object.append(text)

df['xmin'], df['ymin'], df['xmax'], df['ymax'], df['Object']  = xmin,ymin,xmax,ymax,Object 
df = df[df.Object != " "]

print(df)

#df.to_csv('test550_scratchpart2' + '.csv' ,index = False)
cv2.imwrite('test550_scratchpart2' + '.jpg', img)

cv2.imwrite(output, img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



