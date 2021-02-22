# Detection in Crowded Scenes: One Proposal, Multiple Predictions

This is the pytorch re-implementation of the paper "[Detection in Crowded Scenes: One Proposal, Multiple Predictions](https://openaccess.thecvf.com/content_CVPR_2020/html/Chu_Detection_in_Crowded_Scenes_One_Proposal_Multiple_Predictions_CVPR_2020_paper.html)", https://arxiv.org/abs/2003.09163, published in CVPR 2020.

Our method aiming at detecting highly-overlapped instances in crowded scenes.

The key of our approach is to let each proposal predict a set of instances that might be highly overlapped rather than a single one in previous proposal-based frameworks. With this scheme, the predictions of nearby proposals are expected to infer the **same set** of instances, rather than **distinguishing individuals**, which is much easy to be learned. Equipped with new techniques such as EMD Loss and Set NMS, our detector can effectively handle the difficulty of detecting highly overlapped objects.

The network structure and results are shown here:

<img width=60% src="https://github.com/Purkialo/images/blob/master/CrowdDet_arch.jpg"/>
<img width=90% src="https://github.com/Purkialo/images/blob/master/CrowdDet_demo.jpg"/>

# Citation

If you use the code in your research, please cite:
```
@InProceedings{Chu_2020_CVPR,
author = {Chu, Xuangeng and Zheng, Anlin and Zhang, Xiangyu and Sun, Jian},
title = {Detection in Crowded Scenes: One Proposal, Multiple Predictions},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

# Run
1. Requirements:
    * python 3.6.8, pytorch 1.5.0, torchvision 0.6.0, cuda 10.1

2. CrowdHuman data:
    * CrowdHuman is a benchmark dataset containing highly overlapped objects to better evaluate wether a detector can better handle crowd scenarios. The dataset can be downloaded from http://www.crowdhuman.org/. The path of the dataset is set in `config.py`.

3. Steps to run:
    * Step1:  training. More training and testing settings can be set in `config.py`.
	```
	cd ROOT_DIR/model/DETECTOR_NAME/OWNER_NAME/project
	#cd tools
	#python3 train.py -d NUM_GPUS
	```
    
	* Step2:  testing. If you have four GPUs, you can use ` -d 0-3 ` to use all of your GPUs.
			  The result json file will be evaluated automatically.
	```
	#cd tools
	cd ROOT_DIR/model/DETECTOR_NAME/OWNER_NAME/project
	python3 test_net.py -d 0-NUM_GPUS -r 30 -e 50
	```
    
	* Step3:  evaluating json, inference one picture and visulization json file. All of the value correpsponding the different evalutation metric will be calculated and be saved in a log file
	```
	cd ROOT_DIR/model/DETECTOR_NAME/OWNER_NAME/project
	python3 demo.py
	#cd tools
	#python3 eval_json.py -f your_json_path.json
	#python3 inference.py -md rcnn_fpn_baseline -r 40 -i your_image_path.png 
	#python3 visulize_json.py -f your_json_path.json -n 3
	```

# Models

We use MegEngine in the research (https://github.com/megvii-model/CrowdDetection), this proiect is a re-implementation based on Pytorch.
<!-- We use pre-trained model from Detectron2 Model Zoo: https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl. (or [R-50.pkl](https://drive.google.com/open?id=1qWAwY8QOhYRazxRuIhRA55b8YDxdOR8_)) -->
We use pre-trained model from [MegEngine Model Hub](https://megengine.org.cn/model-hub) and convert this model to pytorch. You can get this model from [here](https://drive.google.com/file/d/1lfYQHC63oM2Dynbfj6uD7XnpDIaA5kNr/view?usp=sharing).
These models can also be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1U3I-qNIrXuYQzUEDDdISTw)(code:yx46).
| Model | Top1 acc | Top5 acc |
| --- | --- | --- |
| ResNet50 | 76.254 | 93.056 |
