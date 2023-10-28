# Perception for CAVs Under Adverse Weather Conditions

* This project aims to evaluate the resilience of detection frameworks deployed in connected autonomous vehicles under adverse weather conditions. The experiments involve fog simulation on the [OPV2V Dataset](https://mobility-lab.seas.ucla.edu/opv2v/).
* The [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) framework was utilised.
* The [PointPillars](https://arxiv.org/abs/1812.05784) 3D detection backbone was utilised.
* The fusion of feature maps from different vehicles was achieved using the [Spatial-wise Adaptive Feature Fusion](https://arxiv.org/abs/2208.00116) (S-AdaFusion) method.

## Installation
For detailed instructions on setup and installation, please refer to the [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) repository.

## Training 
`python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml` 

## Evaluation 
Before you run the following command, first make sure the validation_dir in config.yaml under your checkpoint folder refers to the testing dataset path (**opv2v_data_dumping/test** or **opv2v_data_dumping/test_culver_city**). To test the pretrained model run:  

```python opencood/tools/inference.py --model_dir opencood/logs/pretrained_model2_intermediate_point_pillar_SAdaFusion_inside_the_backbone_pillar_size_02 --fusion_method 'intermediate'```


## Results of the proposed framework on OPV2V Dataset

#### Results on Clear Weather Conditions
| AP@0.7  | AP@0.5  | AP@0.3  |
| ------- |:-------:|:-------:|
|  0.877  |  0.92   |  0.93   |

 ##### Visualized Results
 The green boxes represent the ground truth bounding boxes, while the red ones represent the detected vehicles in the scene. The vehicles inside the orange square are the ones detected because of cooperative perception.
 
|<img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/GroudTruthBoundingBoxes.png" width="600" height="200">| <img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/cav1_ego.png" width="600" height="200">|
|:--:|:--:|
| *Ground Truth Bounding Boxes of all the vehicles on the current scene* |*Ground trouth bounding boxes for all the vehicles within the field of view of Cav1*|

|<img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/cav2.png" width="600" height="200">| <img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/cav3.png" width="600" height="200">|
|:--:|:--:|
| *Ground trouth bounding boxes for all the vehicles within the field of view of Cav2* |*Ground Trouth Bounding Boxes of all the vehicles within the field of view of Cav3*|

|<img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/NoFusion.png" width="600" height="200">| <img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/NotedDetectedVehicles.png" width="600" height="200">|
|:--:|:--:|
| *Results of detection without Cooperative Perception* |*Results of detection with Cooperative Perception*|

#### Results on Adverse Weather Conditions
* The experiments are conducted under two distinct conditions: clear weather conditions and fog simulation. 
* The density of the fog is quantified in terms of driver visibility, measured in meters. 

| Visibility    |Default Towns |  culver City |
| ------------- |:------------:|:------------:|
|     (m)      |    AP@0.7    |    AP@0.7    |
| Clear Weather |    0.88      |    0.81      |
|      140      |    0.74      |    0.63      |
|      120      |    0.73      |    0.61      |
|      100      |    0.72      |    0.59      |
|      80       |    0.70      |    0.55      |
|      60       |    0.66      |    0.49      |
|      40       |    0.61      |    0.41      |

##### Visualized Results
A comparison between the same scenario in clear weather and under varying levels of fog. The colour change of the points indicates the reduced intensity due to the fog. The blue colour indicates low intensity.

|<img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/Original.png" width="600" height="200">| <img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/vis_120.png" width="600" height="200">|
|:--:|:--:|
| *Original scenario from the OPV2V dataset.* |*Added fog with V = 120m.*|

|<img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/vis_80.png" width="450" height="200">| 
|:--:|
| *Added fog with V = 80m.* |


##### Detection on fog simulation

Examples of cooperative perception on the OPV2V dataset. The ground truth bounding boxes are shown in green, while the predicted bounding boxes are shown in red. The blue circles indicate vehicles that are occluded and would not be detected in single vehicle perception.

|<img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/og__.png" width="600" height="200">| <img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/vis60.png" width="600" height="200">|
|:--:|:--:|
| *Original point cloud from the OPV2V dataset.* |*Detected vehicles in fog with V = 60m.*|


##### Comparison with other SOTA methods
Results of different methods of cooperative perception and single vehicle perception under fog simulation. On the left are depicted the results from the Default Town, while on the right the results from the Culver City.

|<img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/default_towns_new.png" width="400" height="300">| <img src="https://github.com/vvrgroup/TsakmakopoulouThesis/blob/main/images/culver_city_new.png" width="400" height="300">|
|:--:|:--:|



## Acknowledgements
* Xu, R., Xiang, H., Xia, X., Han, X., Li, J., & Ma, J. (2022, May). Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-to-vehicle communication. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 2583-2589). IEEE.
* Qiao, D., & Zulkernine, F. (2023). Adaptive feature fusion for cooperative perception using lidar point clouds. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1186-1195).
