# RoboGround
Code &amp; data for "RoboGround: Robotic Manipulation with Grounded Vision-Language Priors" (CVPR 2025) [[Project Page]](https://robo-ground.github.io/) [[Paper]](https://arxiv.org/abs/2504.21530)


## 🔨 Environment Setup

- Clone the repository with submodules
    ```bash
    git clone --recurse-submodules https://github.com/ZzZZCHS/RoboGround.git
    cd RoboGround
    ```

- Create and activate the Conda environment
    ```bash
    conda env create -f roboground.yml
    conda activate roboground
    ```

- Install dependencies
    ```bash
    pip install -e robosuite
    pip install -e robomimic
    pip install -e robocasa

    pip install PyOpenGL==3.1.9
    ```

- Set up the private macro file for robocasa
    ```bash
    cd robocasa
    python robocasa/scripts/setup_macros.py
    ```

## Data

- Download and extract assets

    Download the following asset files from from [RoboGround_Data](https://huggingface.co/datasets/ZzZZCHS/RoboGround_Data/tree/main):
    - `robosuite_assets.tar.gz`
    - `robocasa_assets.tar.gz`
    
    Then, extract them into their respective directories:
    ```bash
    tar -xzvf /path/to/robosuite_assets.tar.gz -C robosuite/robosuite/models/
    tar -xzvf /path/to/robocasa_assets.tar.gz -C robocasa/robocasa/models/
    ```

- Download and extract generated data
    Download the generated data files (all files with extensions .zip, .z01, .z02, .z03) from [RoboGround_Data](https://huggingface.co/datasets/ZzZZCHS/RoboGround_Data/tree/main). 
    
    Next, use 7-Zip to unzip the files:
    - First, download and extract 7-Zip:
        ```bash
        wget https://www.7-zip.org/a/7z2409-linux-x64.tar.xz
        mkdir 7zip
        tar -xvf 7z2409-linux-x64.tar.xz -C 7zip
        ```
        **Reminder**: If you can directly install 7-Zip using your system's package manager (e.g., `sudo apt install p7zip-full` on `Ubuntu`), you can skip the manual installation process above. In this case, make sure to modify `./7zip/7zz` in `scripts/unzip_files.sh` to use the default `7z` command.

    - Then, unzip the downloaded data files using the provided script:
        ```bash
        bash scripts/unzip_files.sh /path/to/RoboGround_Data
        ```

    - After successful extraction, you can remove the .zip and .z* files:
        ```bash
        rm /path/to/RoboGround_Data/*.z*
        ```

- Visualize demonstrations:
    ```bash
    cd robomimic
    bash robomimic/scripts/run_visualization.sh /path/to/TASK_NAME.hdf5
    ```

- Data generation:

    We created a custom dataset based on [robocasa's demonstrations](https://robocasa.ai/docs/use_cases/downloading_datasets.html) by introducing object distractors and generating new instructions. 

    - First, add objects and generate appearance-based instructions:
        ```bash
        cd robomimic
        python robomimic/scripts/generate_demos.py \
            --dataset /path/to/hdf5_data \  # Path to the original HDF5 dataset
            --n 3000 \  # Number of demonstrations to process (use a small number for debugging)
            --camera_height 512 --camera_width 512 \  # The size of observation images/masks
            --save_new_data --save_obs \  # Enable saving of augmented data and observations
            --write_gt_mask \  # Save ground-truth masks for target objects and placement areas
            --write_video \  # Save video visualizations if needed
            --use_actions  # Replay original robot actions
        ```

    - Once new demonstrations are generated, create corresponding spatial and commonsense instructions. Update the `file_paths` and `output_dir` variables in the following scripts:
        ```bash
        cd data_gen/gpt
        python generate_spatial_instructions.py
        python generate_common_instructions.py
        ```

## Training & Evaluation
Using the generated dataset, we first train a grounded vision-language model (VLM) to detect target objects and placement areas. For implementation details, refer to our groundingLMM [repository](https://github.com/ZzZZCHS/groundingLMM).

Next, we train a robot policy using the GR-1 framework. Full implementation details can be found in our GR-1 [repository](https://github.com/ZzZZCHS/GR1).

## TODO

- [x] Data release.
- [x] Code and instruction for data generation.
- [x] Code and instruction for model training&evaluation.

## 😊 Acknowledgement

Thanks to the open source of the following projects:
[robocasa](https://github.com/robocasa/robocasa/tree/main), [robomimic](https://github.com/ARISE-Initiative/robomimic), [robosuite](https://github.com/ARISE-Initiative/robosuite/tree/master), [GLaMM](https://github.com/mbzuai-oryx/groundingLMM)