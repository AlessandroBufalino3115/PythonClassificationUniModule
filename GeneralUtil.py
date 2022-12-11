from PIL import Image
import os
from os import listdir
import random

testList = [ 8 ,15 , 20 ,13 ,16, 20 ,19, 13, 20, 20, 14 ,16]  
valList = [  8 , 15, 20 ,13, 17 ,20, 20, 12, 20 ,20, 13 ,17]
trainList = [65, 75, 87 ,66, 78, 87 ,80, 47 ,85 ,96, 68, 72]

sizeList = [81,105,127,92,111,127,119,72,125,136,95,105]



# num = 0
# for i in range(len(sizeList)):
#     num += testList[i]

# print(num)




# get ratio of pics
# print("true ratio = " + str(round(15/105,5)) + "  " + str(round(15/105,5))  + "  "  +  str(round(75/105,5)))

# for i in range(len(sizeList)):
#     print("---------------------------------")
#     print("ratio for index " + str(i))
#     print("val " + str(round(valList[i]/sizeList[i], 5)))
#     print("test " + str(round(testList[i]/sizeList[i],5)))
#     print("train " + str(round(trainList[i]/sizeList[i], 5)))





#randomly rotate all the images, take out black background too

folder_dir = r"C:\Users\Alex\Desktop\Uni work\Year 3\AI for Creative Technologies\DatasetRandom\Train"
for folders in os.listdir(folder_dir):
    newfolderName = folder_dir+ "\\" + folders
    for images in os.listdir(newfolderName):
    
        picName = newfolderName + "\\" + images
    
        rotate = random.randint(0, 360)
        img = Image.open(picName)
        img = img.rotate(rotate, expand=True) 
    
    
        img = img.convert("RGBA")
    
        newData = []
        datas = img.getdata()
    
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
    
        img.putdata(newData)
        img.save(picName)
        print(images)
    


folder_dir = r"C:\Users\Alex\Desktop\all_images_png (1)\all_images_png"
for images in os.listdir(folder_dir):
    picName = folder_dir + "\\" + images
    img = Image.open(picName)
    img = img.convert("RGB")
    img.save(picName)
    print(images)
        
# convert all iages to have 3 Channel for GAN RGB
