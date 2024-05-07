# isaac-sim-pick-place

## Environment Setup

### Setting Step by Step([Detail](https://gistailab.notion.site/9-5b2bb677af4a4b729ac842c012ec073f?pvs=4))
#### 1. Download Isaac Sim
 - Dependency check
    - Ubuntu
      - Recommanded: 20.04 / 22.04
      - Tested on: 20.04
    - NVIDIA Driver version
      - Recommanded: 525.60.11
      - Minimum: 510.73.05
      - Tested on: 510.108.03 / 
 - [Download Omniverse](https://developer.nvidia.com/isaac-sim)
 

#### 2. Python

- Check [Python Environment Installation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda)
 
- move to `Isaac-sim` installed path
   ```
   isaac_path = /home/${USER}/.local/share/ov/pkg/isaac_sim-{version}
   cd isaac_path
   ```

- Create env create
   ```
   conda env create -f environment.yml
   conda activate isaac-sim
   ```

- Setup environment variables so that Isaac Sim python packages are located correctly
   ```
   source setup_conda_env.sh
   ```

- Install requirment pakages
   ```
   pip install -r requirements.txt
   ```

- Install pycocotools
   ```
   conda install -c conda-forge pycocotools
   ```

####


### Docker
- Install Init file
  ```
  wget https://raw.githubusercontent.com/gist-ailab/AILAB-isaac-sim-pick-place/main/dockers/init_script.sh
  zsh init_script.sh
  ```
