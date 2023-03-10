# Scene Reinvention
This is an optimization-based framework for rearranging indoor furniture to accommodate human-robot co-activities better. The rearrangement aims to afford sufficient accessible space for robot activities without compromising everyday human activities.

Link to Paper; 
[Link to Website](https://sites.google.com/view/coactivity/home)

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
python3 scripts/process_posnet.py -r bedroom -c -p -o
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -r --room       |	bedroom           |define room category to be iterated (options: living, office, bedroom)
| -i  --images         | False           |Generate top-down images for the rooms as it collects data.
| -g --graph 	       |	False	            |Generate scene gragh visualization as it collects data.
| -c --collect		       | False           | Collect object poses from SUNCG, needed for -p and -o
| -p --pos		       | False           | Learn the distribution of the objects' relative poses.
| -o --occ		       | False           | Learn the objects' spatial co-occurrence.


The images and pickle files are generated under ``./suncg/data/graphs/``. Both collecting and learning can take several hours.

## Scene Rearrangement
You can perform scene rearrangement using this script. There are some rooms already compiled in `rooms/suncg/`, but you may copy any room from `suncg/data/graphs/` and paste into `rooms/` if you have previously completed the *Collecting Data* step.

Example, performing rearrangement with bullet GUI on:
```bash
python3 scripts/optimize_suncg_room.py --room living --idx 381 --GUI
```

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --room       |	living           |define room category.
| --idx         | ""           |define the ID of the selected room category.
| --GUI 	       |	False	            |define whether the GUI powered by Pybullet will be displayed. This will show the optimization process in real time.
| --images		       | False           |define whether buffer images will be generated.
| --robot         | dingo           | define the robot's base size (options: dingo, husky)

When a room is rearranged for the first time, the semantic relations of human preference are queried from [Conceptnet](https://conceptnet.io/). This may take some time. Temporary files can be found under ``./buffer``.

The pickle results are generated under ``./results``. The results are distingushed by the order of functional groups and search method. The final results will have the ID of "-1". For example: ``bedroom_473_layered_-1_cma`` is the results for all functional groups after Covariance Matrix Adaptation Evolution Strategy, while ``bedroom_473_layered_3`` is the results for the third functional groups after Adaptive Simulated Annealing.


## Scene Visualization
You can visualize the environments by generating an image.

Example 1, viewing the original SUNCG room as a gif:
```bash
python3 scripts/visualize_results.py -i living_12 -g
```

Example 2, viewing an optimized room fitted for human robot co-activity:

```bash
python3 scripts/visualize_results.py -r bedroom_473_layered_-1_cma
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --room_id       |	""           |the room_id to be shown, ignored if there is no input.
| -r  --results         | ""           |the optimization result to be shown, ignored if there is no input.
| -g --gif 	       |	False	            |generate rotating gifs for output.
| -s --sequence		       | False           | generate optimization sequence image, only works with --results input, not compatible with --gif
| -p --resolution 		           | 700             | define the image resolution.
| -b, --accessible	        | False           | overlay the output scene with (B)lue accessible space.
| -l --path	         | False             | overlay the output scene with robot path (L)ine.
| -a --interaction         | False           | visualize the pseudo-inter(A)ction function for each object.
| -c --color         | False           | (C)olor-code the objects by whether they are accessible to the robot.
| --robot         | dingo           | define the robot's base size (default: dingo, options: dingo, husky), needed for -b, -l, -a, and -c.


The images are generated under ``./images/``

