# CGD_OpenVINO_Demo
Efficient Inference and Quantization of CGD for Image Retrieval with OpenVINO

This demo is base on [CGD](https://github.com/leftthomas/CGD/tree/master): A PyTorch implementation of CGD based on the paper [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663v3).

### Setup Environment
```bash 
conda create -n CGD python=3.8
pip install openvino==2023.0.1 openvino-dev[pytorch,onnx]==2023.0.1 nncf==2.5.0 torch==2.0.1
```

### Prepare dataset based on [Standard Online Products](http://cvgl.stanford.edu/projects/lifted_struct)
```bash
sudo mkdir -p /home/data/sop
sudo chmod -R 777 /home/data/sop
python data_utils.py --data_path /home/data
```

### Downlaod pre-trained Pytorch Model [ResNet50(SG) trained on SOP dataset](https://github.com/leftthomas/CGD/tree/master#sop)
```
cp <PATH/TO/DIR>/sop_uncropped_resnet50_SG_1536_0.1_0.5_0.1_128_model.pth results
cp <PATH/TO/DIR>/sop_uncropped_resnet50_SG_1536_0.1_0.5_0.1_128_data_base.pth results
```

### Verify Pytorch FP32 Model Image Retrivial Results
```bash 
python test.py --query_img_name /home/data/sop/uncropped/281602463529_2.JPG \
               --data_base sop_uncropped_resnet50_SG_1536_0.1_0.5_0.1_128_data_base.pth  \
               --retrieval_num 8
```
#### Pytorch FP32 Model Retrieval Results
![Pytorch FP32 Model Retrieval Results](results/pytorch_retrieval_result.png)
The leftmost query image serves as input to retrieve the 8 most similar image from the database, where the green bounding box means that the predicted class match the query image class, while the red bounding box means a mismatch of image class. Therefore, the retrieved image can be further filtered out with class information.

### Run NNCF PTQ for quantization
```
mkdir -p models
python run_quantize.py
```
Generated FP32 ONNX model and FP32/INT8 OpenVINO™ model will be saved in the `models` directory. Besides, we also store evaluation results of OpenVINO™ FP32/INT8 model as a Database in the `results` directory respectively. The database can be directly used for image retrieval via input query image.

### Verify OpenVINO FP32 Model Image Retrivial Results
```
python test.py --query_img_name /home/data/sop/uncropped/281602463529_2.JPG \
               --data_base ov_fp32_model_data_base.pth  \
               --retrieval_num 8
```

### Verify OpenVINO INT8 Model Image Retrivial Results
```
python test.py --query_img_name /home/data/sop/uncropped/281602463529_2.JPG \
               --data_base ov_int8_model_data_base.pth  \
               --retrieval_num 8
```
#### Pytorch FP32 Model and OpenVINO FP32/INT8 Retrieval Results with Same Query Image
![Pytorch FP32 Model and OpenVINO FP32/INT8 Retrieval Results](results/pytorch_openvino_retrieval_result.png)
The Pytorch and OpenVINO™ FP32 retrieved images are the same. Although the 7th image of OpenVINO™ INT8 model results is not matched with FP32 model, it can be further filtered out with predicted class information.
