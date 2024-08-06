import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def download():    
    base_path = './ckpt_models'
    create_directory(base_path)          
    os.system(f'git clone https://code.openxlab.org.cn/tanshuai0219/EDTalk.git {base_path}')
    os.system(f'cd {base_path} && git lfs pull')   

if __name__ == "__main__":
    download()
