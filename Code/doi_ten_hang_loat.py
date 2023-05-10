import os

# [ Đổi tên hàng loạt ]

def rename_file(folder_path,change_str, file_ext, number):
    for file in os.listdir(folder_path):
        old_file_path= os.path.join(folder_path,file) 
        if os.path.isfile(old_file_path):
            old_file_name=os.path.basename(old_file_path)
            new_file_name= change_str+str(number)+file_ext
            new_file_path=os.path.join(folder_path,new_file_name)        
            os.rename(old_file_path,new_file_path)
            number+=1

folder_path ="D:\Desktop\TGMT\DA_TGMT\Images\Tram\T" # Đường dẫn thư mục chứa file
change_str="Génhin_" # tên tùy chỉnh
file_extension=".txt" #
number = 1 # số bắt đầu ở đuôi file (vd: abc_1.txt, abc_2.txt)
rename_file(folder_path,change_str, file_extension, number)
