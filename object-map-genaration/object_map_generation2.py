import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from numpy import long


import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


images = load_images_from_folder("data/input2")

# df = pd.read_csv("image/4.csv")

for j in range(len(images)):

    path = "data/cards/raw/c" + str(j) + ".jpg"
    df = process_text_analysis(path)

    # r = [, 0, 0, 0, 0, "", "o"]
    new_df = pd.DataFrame( columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    label = "o"
    text = ""
    xmin = 0
    xmax = 0
    ymin = df.loc[0, "ymin"]
    ymax = df.loc[0, "ymax"]
    image = Image.open(path)
    imgWidth, imgHeight = image.size

    for i in range(len(df) - 1):
        space = abs(df.loc[i, "xmax"] - df.loc[i + 1, "xmin"]) * 100
        # width = df.loc[i , "xmax"]-df.loc[i , "xmin"]
        # numberOfChar = len(df.loc[i,"Object"])
        # per = width / numberOfChar
        if space == 0:
            continue

        Height = max((ymax - ymin), (df.loc[i, "ymax"] - df.loc[i, "ymin"])) * 100
        rate = (Height / space)
        text += (str)(df.loc[i, "Text"])
        text += " "
        # line1 = df.loc[i, "ymax"] * 100
        # line2 = df.loc[i+1, "ymax"] * 100
        # dif = abs(line1-line2)
        # print(line1, line2, dif, text)
        if space < Height:
            ymin = min(ymin, df.loc[i, "ymin"])
            ymax = max(ymax, df.loc[i, "ymax"])
        else:
            xmax = df.loc[i, "xmax"]
            # v2 = long(xmin)
            new_df.loc[len(new_df.index)] = [long(xmin*imgWidth), long(ymin*imgHeight), long(xmax*imgWidth), long(ymin*imgHeight), long(xmax*imgWidth), long(ymax*imgHeight), long(xmin*imgWidth), long(ymax*imgHeight), text, ""]
            draw = ImageDraw.Draw(image)

            left = imgWidth * xmin
            top = imgHeight * ymin
            draw.rectangle([left, top, left + (imgWidth * (xmax - xmin)),
                            top + (imgHeight * (ymax - ymin))], outline='red')
            text = ""
            ymin = df.loc[i + 1, "ymin"]
            ymax = df.loc[i + 1, "ymax"]
            xmin = df.loc[i + 1, "xmin"]

    text += (str)(df.loc[i + 1, "Text"])
    xmax = df.loc[i + 1, "xmax"]
    new_df.loc[len(new_df.index)] = [long(xmin*imgWidth), long(ymin*imgHeight), long(xmax*imgWidth), long(ymin*imgHeight), long(xmax*imgWidth), long(ymax*imgHeight), long(xmin*imgWidth), long(ymax*imgHeight), text, ""]

    left = imgWidth * xmin
    top = imgHeight * ymin
    draw.rectangle([left, top, left + (imgWidth * (xmax - xmin)),
                    top + (imgHeight * (ymax - ymin))], outline='green')

    image.show()
    print(j)
    image.save("data/box_images2/c"+str(j) + ".jpg")
    new_df = new_df.drop(0)
    new_df.to_csv("data/cards/box/c"+str(j) + ".csv",index=False, header=False,)
