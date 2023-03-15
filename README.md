# Rearrange Indoor Scenes for Human-Robot Co-Activity
![18](https://img.shields.io/badge/Ubuntu-18.04-blue) ![20](https://img.shields.io/badge/Ubuntu-20.04-blue) ![22](https://img.shields.io/badge/Ubuntu-22.04-blue) ![py](https://img.shields.io/badge/Python-3.8+-green)

This is an optimization-based framework for rearranging indoor furniture to accommodate human-robot co-activities better. The rearrangement aims to afford sufficient accessible space for robot activities without compromising everyday human activities.

\[  [ arXiv](https://arxiv.org/abs/2303.05676)   \]
\[  [ Website](https://sites.google.com/view/coactivity/home) \]

# Results
## Rooms suited for both human preference and robot preference
| <img src= "./images/bedroom_669_layered_-1_cma_-1.gif" > | <img src= "./images/living_107_layered_-1_cma_-1.gif"  > | 
| --- | --- |
| <img src= "./images/living_445_layered_-1_cma_-1.gif"> |  <img src= "./images/office_547_layered_-1_cma_-1.gif" w> |


# Setup

## System Requirement
Tested on clean ubuntu 18.04 & 20.04 with Python 3.8+. 

## Install
Please set up a virtual environemnt first
```bash
git clone --recursive git@github.com:Rayckey/scene_coactivity.git scene_coactivity
cd scene_coactivity
virtualenv coactivity
source coactivity/bin/activate
pip3 install -r requirements.txt # or pip if you don't have python 2 installed
```

if you run into an error in installing requirements related to lapack/blas resources, you may need to install the coresponding packages first using:

```bash
sudo apt-get install -y libopenblas-dev gfortran
```

SunCG dataset is needed for visualization, unforturantly it is not avaliable anymore.

If you have it downloaded, unzip all the folders in a location of your choice, then modify the following paths in ``configs/dataset_paths.yaml``

```yaml
MODEL_CATEGORY_FILE: ....../metadata/ModelCategoryMapping.csv
MODELS_FILE: ....../metadata/models.csv
SUNCG_PATH : ...... # root of all unzipped folders
MODEL_PATH : ....../object
```

# Running

## Learning Human Preference
This script collects pre-compiled SUNCG room data and learns the human preference. The learned binaries are already included in this git repo.

Example of collecting the bedroom data and learning the spatial co-occurrence:
```bash
python3 scripts/learn_human_preference.py -r bedroom -c -p -o
```

Arguments: 
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -r --room       |	bedroom           |define room category to be iterated (options: living, office, bedroom)
| -i  --images         | False           |generate top-down images for the rooms as it collects data.
| -g --graph 	       |	False	            |generate scene gragh visualization as it collects data.
| -c --collect		       | False           |collect object spatial information from SUNCG, needed for -p and -o
| -p --pos		       | False           |learn the distribution of the objects' relative poses.
| -o --occ		       | False           |learn the objects' spatial co-occurrence.


The images and pickle files are generated under ``./suncg/data/graphs/``. Option -c, -p, and -o can take several hours.

## Scene Rearrangement
You can perform scene rearrangement using this script. There are some rooms already compiled in `rooms/suncg/`, but you may copy any room from `suncg/data/graphs/` and paste it into `rooms/` if you have previously completed the *Learning Human Preference* step.

Example, performing rearrangement on `living_381` with bullet GUI on:
```bash
python3 scripts/rearrange_suncg_scene.py --room living --idx 381 --GUI
```
Arguments:
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --room       |	living           |define room category.
| --idx         | None          |define the room ID from the selected room category.
| --GUI 	       |	False	            |define whether the GUI powered by Pybullet will be displayed. This will show the optimization process in real time.
| --images		       | False           |define whether debug images will be generated.
| --robot         | dingo           | define the robot's base size (options: dingo, husky)

When a room is rearranged for the first time, relevant object semantic relations are queried from [Conceptnet](https://conceptnet.io/) as a part of human preference. This may take some time. Temporary files can be found under ``./buffer``.

Optimization uses the weights defined in ``./configs/weights.yaml``.

The pickle results are generated under ``./results``. The results are distingushed by the order of functional groups and search method. The final results will have the ID of "-1". For example: ``bedroom_473_layered_-1_cma`` is the results for all functional groups after Covariance Matrix Adaptation Evolution Strategy, while ``bedroom_473_layered_3`` is the results for the third functional groups after Adaptive Simulated Annealing.


## Scene Visualization
You can visualize the scenes by generating images.

Example 1, viewing the original SUNCG room as a rotating gif:
```bash
python3 scripts/scene_visualization.py -i living_12 -g
```

Example 2, viewing an optimized room fitted for human robot co-activity in a single image:

```bash
python3 scripts/scene_visualization.py -r bedroom_473_layered_-1_cma
```

Arguments: 
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --room_id       |	""           |the original room_id to be shown, ignored if there is no input.
| -r  --results         | ""           |the optimized result to be shown, ignored if there is no input.
| -g --gif 	       |	False	            |generate rotating gifs for output.
| -s --sequence		       | False           |generate optimization sequence image, only works with --results input, not compatible with --gif
| -p --resolution 		           | 700             | define the image resolution.
| -b, --accessible	        | False           | overlay the output scene with (B)lue accessible space.
| -l --path	         | False             | overlay the output scene with robot path (L)ine.
| -a --interaction         | False           | visualize the pseudo-inter(A)ction function for each object.
| -c --color         | False           | (C)olor-code the objects by whether they are accessible to the robot.
| --robot         | dingo           | define the robot's base size (default: dingo, options: dingo, husky), needed for -b, -l, -a, and -c when displaying original SUNCG room


The images are generated under ``./images/``

