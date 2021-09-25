import os, sys, json, argparse
from pathlib import Path
import gdown
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

project_dir = Path(os.path.realpath(__file__)).parent
data_dir = os.path.join(str(project_dir), 'data', 'physionet', 'afdb')
src_dir = os.path.join(project_dir, 'src')

# add project modules
sys.path.insert(1, src_dir)

# import eda and etl
from src import eda, etl


def main():
    parser = argparse.ArgumentParser(description='AFib Detection Project Setup')
    parser.add_argument('--all', action='store_true', default=False,
                        help='Performs all actions below')
    parser.add_argument('--download-data', action='store_true', default=False,
                        help='Loads data from Physionet Database')
    parser.add_argument('--download-models', action='store_true', default=False,
                        help='Downloads pretrained models')
    parser.add_argument('--build', action='store_true', default=False,
                        help='Builds folder structure and label mappings')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Evaulates models in models folder')
    args = parser.parse_args()
    
    make_config()
    
    if args.all:
        download_dataset()
        download_pretrained_models()
        build()
        evaluate()
        return
    
    if args.download_data:
        download_dataset()
    if args.download_models:
        download_pretrained_models()
    if args.build and (args.download_data or os.path.exists(data_dir)):
        build()
    if args.eval:
        evaluate()

def evaluate():
    if 'eval_all' not in sys.modules:
        from src import eval_all
    eval_all.main()

def make_config():
    config = {
        'project_dir': str(project_dir),
        'data_dir': data_dir
    }
    config_outpath = os.path.join(project_dir, 'config.json')
    with open(config_outpath, 'w') as config_file:
        json.dump(config, config_file)
        
def download_dataset():
    print('Downloading Physionet Data...')
    url = 'https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip'
    physionet_resp = urlopen(url)
    physionet_zip = ZipFile(BytesIO(physionet_resp.read()))
    
    print('Unpacking Data...')
    dest_folder = os.path.join('data', 'physionet')
    physionet_zip.extractall(dest_folder)
    os.rename(os.path.join(dest_folder, 'files'), os.path.join(dest_folder, 'afdb'))

def download_pretrained_models():
    url = 'https://drive.google.com/uc?id=1lDG_HlQ8aQN0ttkfQkcmR2eswxLvymWF'
    output = 'models.zip'
    
    model_zip_md5 = '9238871a80bd95522ea45c01df351eba'
    gdown.cached_download(url, output, md5=model_zip_md5)
    
    ZipFile(output).extractall()
    
def build():
    os.mkdir(os.path.join(str(project_dir), 'data', 'cleaned'))
    os.mkdir(os.path.join(str(project_dir), 'data', 'temp'))
    
    eda.generate_label_maps()
    etl.generate_train_test_split_map()
    etl.explode_split_map()
    
        
if __name__ == '__main__':
    main()