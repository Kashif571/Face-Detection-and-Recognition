import os

#Current Directory
folderaddress =  os.chdir(r'E:/Kashif Iftikhar/University/FYP/Code/Python/PR New/Test Data/Hassan Zamir')

i=1

for file in os.listdir(folderaddress):
    #For Text with Number
    # new_file_name =  "Image{}.jpg".format(i)

    #For Only Numbers
    new_file_name =  "{}.jpg".format(i)
    
    #Changing Filing Name
    os.rename(file, new_file_name)
    i=i+1