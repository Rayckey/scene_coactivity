# Scene Reinvention
This is an optimization-based framework for rearranging indoor furniture to accommodate human-robot co-activities better. The rearrangement aims to afford sufficient accessible space for robot activities without compromising everyday human activities.

Link to Paper; 
[Link to Website](https://sites.google.com/view/coactivity/home)

# Results
<img src= "./images/bedroom_669_layered_-1_cma_-1.gif" width=30% height=30% >
<img src= "./images/living_107_layered_-1_cma_-1.gif" width=30% height=30% >
<img src= "./images/living_445_layered_-1_cma_-1.gif" width=30% height=30% >
<img src= "./images/office_547_layered_-1_cma_-1.gif" width=30% height=30% >


# Setup

## System Requirement
Tested on clean ubuntu 18.04 & 20.04 with Python 3.8+. 

## Install
Please set up a virtual environemnt first
```bash
git clone --recursive git@github.com:Rayckey/coactivity.git scene_coactivity
cd coactivity
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

## Collecting Data
This script collects pre-compiled SUNCG room data and learns the distribution. The learned binaries are already included in this git repo.

Example:
```bash
python3 scripts/process_posnet.py -r bedroom -r -c -p -o
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -r --room       |	bedroom           |define room type to be iterated (options: living, office, bedroom)
| -i  --images         | False           |Generate top-down images for the rooms as it collects data.
| -g --graph 	       |	False	            |Generate scene graghs for the rooms as it collects.data.
| -c --collect		       | False           | Collect furniture positions from SUNCG, needed for -p and -o
| -p --pos		       | False           | Learn the spatial relation.
| -o --occ		       | False           | Learn the co-occurance relation.


The images and pickle files are generated under ``./suncg/data/graphs/``

## Scene Reinvention
You can perform scene reinvention using this script. There are some rooms already compiled in `rooms/suncg/`, but you may copy any room out of `suncg/data/graphs/` if you have previously completed the *Collecting Data* step.

Example, performing optimization on a room with bullet GUI on:
```bash
python3 scripts/optimize_suncg_room.py --room living --idx 381 --GUI
```

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --room       |	living           |define room type to be iterated.
| --idx         | ""           |define the ID of the selected room type
| --GUI 	       |	False	            |define whether GUI will be used.
| --images		       | False           |define whether buffer images will be generated.
| --robot         | dingo           | define the robot's base size (default: dingo, options: dingo, husky)

The images and pickle files are generated under ``./results``. The results are distingushed by furniture groups and search method. The final results will have the ID of "-1".


## Scene Visualization
You can visualize the environments by generating an image.

Example 1, viewing the original SUNCG room as a gif:
```bash
python3 scripts/visualize_results.py -i living_12 -g
```

Example 2, viewing an optimized results:

```bash
python3 scripts/visualize_results.py -r bedroom_473_layered_-1_cma
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --room_id       |	""           |the room_id to be shown, ignored if there is no input.
| -r  --results         | ""           |the optimization result to be shown, ignored if there is no input.
| -g --gif 	       |	False	            |Generate rotating gifs for output.
| -s --sequence		       | False           | Generate optimization sequence image, only works with --results input, not compatible with --gif
| -p --resolution 		           | 700             | Define the image resolution.
| -b, --reachable	        | False           | Overlay the output scene with (B)lue reachable space.
| -l --path	         | False             | Overlay the output scene with robot path (L)ine.
| -a --affordance         | False           | Overlay the output scene with (A)ffordance.
| -c --color         | False           | (C)olor-code the funiture in the output scene.
| --robot         | dingo           | define the robot's base size (default: dingo, options: dingo, husky),needed for -b, -l, -a, and -c.


The images and pickle files are generated under ``./images/``

