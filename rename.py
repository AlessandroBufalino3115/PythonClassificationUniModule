
import os
 
def main():
   
    folder = r"C:\Users\Alex\Desktop\all_images"
    for count, filename in enumerate(os.listdir(folder)):
       
        # print(filename)
        new_file_name = filename.replace('.jpeg', '.jpg')
 
        os.rename(folder+ "\\"+ filename, folder+ "\\" + new_file_name)
        
        
        # if (filename.__contains__(".png")):
        #     print(filename)
        
 
if __name__ == '__main__':
     
    main()