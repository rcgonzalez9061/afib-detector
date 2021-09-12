import os, json
from pathlib import Path


def main():
    project_dir = Path(os.path.realpath(__file__)).parent
    data_dir = os.path.join(str(project_dir), 'data', 'physionet', 'afdb')
    config = {
        'project_dir': str(project_dir),
        'data_dir': data_dir
    }
    config_outpath = os.path.join(project_dir, 'config.json')
    with open(config_outpath, 'w') as config_file:
        json.dump(config, config_file)

        
if __name__ == '__main__':
    main()