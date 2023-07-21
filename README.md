# CGD_OpenVINO_Demo
Efficient Inference and Quantization of CGD for Image Retrieval with OpenVINO

This demo is base on [CGD](https://github.com/leftthomas/CGD/tree/master): A PyTorch implementation of CGD based on the paper [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663v3).

## Setup Environment
```bash 
conda create -n CGD python=3.8
pip install openvino openvino-dev[pytorch,onnx] nncf
```

## Prepare dataset based on [Standard Online Products](http://cvgl.stanford.edu/projects/lifted_struct)
```bash
python data_util.py --data_path data
```

## Downlaod pre-trained Pytorch Model [ResNet50(SG) trained on SOP dataset](https://github.com/leftthomas/CGD/tree/master#sop)

## Verify Pytorch FP32 Model FP32 Image Retrivial Results
```bash 
python test.py --query_img_name /home/data/sop/uncropped/281602463529_2.JPG --data_base sop_uncropped_resnet50_SG_1536_0.1_0.5_0.1_128_data_base.pth  --retrieval_num 8
```

## Run NNCF PTQ for quantization 
```
python run_quantize.py
```

## Verify OpenVINO FP32 Model Image Retrivial Results
```
python test.py --query_img_name data/sop/uncropped/281602463529_2.JPG --data_base ov_fp32_model_data_base.pth  --retrieval_num 8
```

## Verify OpenVINO INT8 Model Image Retrivial Results
```
python test.py --query_img_name data/sop/uncropped/281602463529_2.JPG --data_base ov_int8_model_data_base.pth  --retrieval_num 8
```
