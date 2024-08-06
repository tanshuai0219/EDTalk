import os
import shutil

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_files(src_dir, dest_dir):
    # 移动ckpts文件夹中的所有文件到新目录
    shutil.move(src_dir + '/ckpts/*', dest_dir + '/ckpts/')
    # 移动EDTalk_lip_pose.pt文件到新目录
    shutil.move(src_dir + '/EDTalk_lip_pose.pt', dest_dir + '/ckpts/')
    # 移动gfpgan文件夹中的所有文件到新目录
    shutil.move(src_dir + '/gfpgan/*', dest_dir + '/gfpgan/weights/')

def download():    
    base_path = './ckpt_models'
    create_directory(base_path)          
    os.system(f'git clone https://code.openxlab.org.cn/tanshuai0219/EDTalk.git {base_path}')
    os.system(f'cd {base_path} && git lfs pull')  
    
    
    create_directory('ckpts')
    create_directory('gfpgan/weights')
    move_files(base_path, '.')

if __name__ == "__main__":
    download()
