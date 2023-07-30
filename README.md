# isaac-sim-pick-place

## Environment Setup

### 1. Download Isaac Sim
 - Dependency check
    - Ubuntu
      - Recommanded: 20.04 / 22.04
      - Tested on: 20.04
    - NVIDIA Driver version
      - Recommanded: 525.60.11
      - Minimum: 510.73.05
      - Tested on: 510.108.03 / 

 - [Download Omniverse](https://developer.nvidia.com/isaac-sim)
 - [Workstation Setup](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html)
 - [Python Environment Installation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda)

### 2. Conda Enviroment Setup
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
