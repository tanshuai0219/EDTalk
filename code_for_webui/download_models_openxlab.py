import os
import shutil

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download():    
    base_path = './ckpt_models'
    create_directory(base_path)          
    os.system(f'git clone https://code.openxlab.org.cn/tanshuai0219/EDTalk.git {base_path}')
    os.system(f'cd {base_path} && git lfs pull')  
    move_files()
    cleanup()

def move_files():
    # 移动 ckpts 文件夹中的文件
    source_ckpts = os.path.join('ckpt_models', 'ckpts')
    dest_ckpts = 'ckpts'
    create_directory(dest_ckpts)
    
    for file in os.listdir(source_ckpts):
        if os.path.exists(os.path.join(dest_ckpts, file)) == False:

            shutil.move(os.path.join(source_ckpts, file), dest_ckpts)
    
    # 移动 EDTalk_lip_pose.pt
    if os.path.exists(os.path.join(dest_ckpts, 'EDTalk_lip_pose.pt')) == False:
        shutil.move(os.path.join('ckpt_models', 'EDTalk_lip_pose.pt'), dest_ckpts)
    
    # 移动 gfpgan 文件夹中的文件
    source_gfpgan = os.path.join('ckpt_models', 'gfpgan')
    dest_gfpgan = os.path.join('gfpgan', 'weights')
    create_directory(dest_gfpgan)
    
    for file in os.listdir(source_gfpgan):
        if os.path.exists(os.path.join(dest_gfpgan, file)) == False:
            shutil.move(os.path.join(source_gfpgan, file), dest_gfpgan)

def cleanup():
    # 删除原始的 ckpt_models 文件夹
    shutil.rmtree('ckpt_models')

if __name__ == "__main__":
    download()
    move_files()
    cleanup()
