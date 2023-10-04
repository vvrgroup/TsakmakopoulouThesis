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


## Results on OPV2V Default Towns

#### Results on Clear Weather Conditions
| AP@0.7  | AP@0.5  | AP@0.3  |
| ------- |:-------:|:-------:|
|  0.877  |  0.92   |  0.93   |

 ##### Visualized Results
 The green boxes represent the ground truth bounding boxes, while the red ones represent the detected vehicles in the scene.
 
|<img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/GroudTruthBoundingBoxes.png" width="600" height="200">| <img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/cav1_ego.png" width="600" height="200">|
|:--:|:--:|
| *Ground Truth Bounding Boxes of all the vehicles on the current scene* |*Ground trouth bounding boxes for all the vehicles within the field of view of Cav1*|

|<img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/cav2.png" width="600" height="200">| <img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/cav3.png" width="600" height="200">|
|:--:|:--:|
| *Ground trouth bounding boxes for all the vehicles within the field of view of Cav2* |*Ground Trouth Bounding Boxes of all the vehicles within the field of view of Cav3*|

|<img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/NoFusion.png" width="600" height="200">| <img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/NotedDetectedVehicles.png" width="600" height="200">|
|:--:|:--:|
| *Results of detection without Cooperative Perception* |*Results of detection with Cooperative Perception*|

#### Results on Adverse Weather Conditions
* The experiments are conducted under two distinct conditions: clear weather conditions and fog simulation. 
* The density of the fog is quantified in terms of driver visibility, measured in meters. 

| Visibility(m) |    AP@0.7     |  
| ------------- |:-------------:|
| Clear Weather |    0.877      |
|      50       |    0.510      |
|      60       |    0.671      |
|      70       |    0.785      |
|      80       |    0.842      |
|      90       |    0.861      |
|      100      |    0.867      |


 ##### Visualized Results
Below we can observe the simulation of fog on the point cloud. The absence of points becomes noticeable as the driver's visibility decreases. The variation in point color indicates an increase in intensity attributed to humidity.

|<img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/Original.png" width="600" height="200">
|:--:|
| *Clear Weather*|

|<img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/Fog_Vis_70.png" width="600" height="200">| <img src="https://github.com/DimitraTsakmakopoulou/Perception-For-Connected-Autonomous-Vehicles-Under-Adverse-Weather-Conditions/blob/main/images/Fog_Vis_50.png" width="600" height="200">|
|:--:|:--:|
| *Visibility = 70m* |*Visibility = 50m (Dense Fog)*|



## Acknowledgements
* Xu, R., Xiang, H., Xia, X., Han, X., Li, J., & Ma, J. (2022, May). Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-to-vehicle communication. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 2583-2589). IEEE.
* Qiao, D., & Zulkernine, F. (2023). Adaptive feature fusion for cooperative perception using lidar point clouds. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1186-1195).
