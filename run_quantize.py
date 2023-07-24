import argparse
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import logging
import model
from utils import recall,ImageReader
import torch
from torch.utils.data import DataLoader
import openvino.runtime as ov
from openvino.runtime import compile_model
from openvino.tools.mo import convert_model

data_path = "/home/data"
data_name = "sop"
test_data_set = ImageReader(data_path, data_name, 'test', 'uncropped')
test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=False, num_workers=8)
eval_dict = {'test': {'data_loader': test_data_loader}}

def validate(model, recalls, model_name, backend="openvino"):
    # obtain feature vectors for all data
    results = {}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []
    for key in eval_dict.keys():
        eval_dict[key]['features'] = []
        features = None
        classes = None
        if backend=="openvino":
            outputs_port = model.outputs
        for inputs, labels in tqdm(test_data_loader, desc='Run inference and caculate accuracy with {} data'.format(key)):
            if backend=="openvino":
                inputs = inputs.numpy()
                outputs = model(inputs)
                features = torch.Tensor(outputs[outputs_port[0]])
                classes = torch.Tensor(outputs[outputs_port[1]])
            elif backend=="pytorch":
                features, classes = model(inputs)
            else:
                print("Invalid backend detected! Validation only support following backend: openvino, pytorch")
            eval_dict[key]['features'].append(features)
        eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)
    acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recalls)
    desc = "Test: "
    for index, rank_id in enumerate(recalls):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
    print(desc)

    data_base = {}
    data_base['test_images'] = test_data_set.images
    data_base['test_labels'] = test_data_set.labels
    data_base['test_features'] = eval_dict['test']['features']
    torch.save(data_base, 'results/{}_data_base.pth'.format(model_name))


    return results, acc_list[0]

def validation_fn(compiled_model: ov.CompiledModel, data_loader: torch.utils.data.DataLoader):
    recall_ids=[1,2,4,8]
    results = {}
    for recall_id in recall_ids:
        results['test_recall@{}'.format(recall_id)] = []
    key = 'test'
    eval_dict = {key: {}}
    eval_dict[key]['features'] = []
    eval_dict[key]['classes'] = []
    outputs_port = compiled_model.outputs
    for inputs in tqdm(data_loader, desc='processing {} data'.format(key)):
        outputs = compiled_model(inputs)
        features = torch.Tensor(outputs[outputs_port[0]])
        eval_dict[key]['features'].append(features)
    eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)
    acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids)
    desc = "Test: "

    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
    print(desc)

    return acc_list[0]


###############################################################################################

import nncf
from openvino.tools import mo
import openvino.runtime as ov

def export_to_openvino_fp32_model(fp32_xml_path: str):
    torch_model = model.Model('resnet50', 'SG', 1536, num_classes=11318)
    torch_model.load_state_dict(
        torch.load("results/sop_uncropped_resnet50_SG_1536_0.1_0.5_0.1_128_model.pth",
                   map_location=torch.device('cpu'))
    )
    torch_model.eval()
    dummy_input = torch.ones(1,3,224,224)

    onnx_model_path = "models/onnx_fp32_model.onnx"
    torch.onnx.export(
        torch_model,
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        dynamic_axes={'input':{0:'N',2:'H',3:'W'}},
        verbose=False
    )

    ov_model = mo.convert_model(onnx_model_path)
    ov.serialize(ov_model, fp32_xml_path)


def validate_ov_model(xml_path: str):
    recalls = [1, 2, 4, 8]
    core = ov.Core()
    ov_model = core.read_model(xml_path)
    compiled_model = core.compile_model(ov_model)
    model_name = os.path.splitext(os.path.basename(xml_path))[0]
    print("Model name: ", model_name)
    results, acc = validate(compiled_model, recalls, model_name, backend="openvino")
    print(results)


def quantize_ov_model(fp32_xml_path: str, int8_xml_path: str):
    core = ov.Core()
    ov_model = core.read_model(fp32_xml_path)

    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    calibration_dataset = nncf.Dataset(test_data_loader, transform_fn)
    ignored_scope = nncf.IgnoredScope(
        names=[
            # bottom
            "/Pow",
            "/Pow_1",
            "561",

            # left
            "/main_modules.0/main_modules.0.1/Pow",
            "/main_modules.0/main_modules.0.1/Pow_1",
            "/main_modules.0/main_modules.0.1/Div",

            # right
            "/main_modules.1/main_modules.1.1/Pow",
            "/main_modules.1/main_modules.1.1/Pow_1",
            "/main_modules.1/main_modules.1.1/Div",

            # right
            "/global_descriptors.1/Pow",
            "/global_descriptors.1/ReduceMean",
            "/global_descriptors.1/Pow_1",
            "/global_descriptors.1/Mul",

            # left
            "/global_descriptors.0/ReduceMean",
        ]
    )
    quantized_model = nncf.quantize(ov_model,
                                    calibration_dataset,
                                    ignored_scope=ignored_scope)

    ov.serialize(quantized_model, int8_xml_path)


if __name__ == '__main__':
    # Export FP32 model
    fp32_xml_path = "models/ov_fp32_model.xml"
    
    export_to_openvino_fp32_model(fp32_xml_path)
    print("Save OpenVINO FP32 model: ", fp32_xml_path)

    # Validate FP32 model
    print('Validate OpenVINO FP32 model:')
    validate_ov_model(fp32_xml_path)

    # Quantize model
    int8_xml_path = "models/ov_int8_model.xml"
    print("Quantize FP32 model")
    
    quantize_ov_model(fp32_xml_path, int8_xml_path)
    print("Save OpenVINO INT8 model: ", int8_xml_path)

    # Validate INT8 model
    print('Validate OpenVINO INT8 model:')
    validate_ov_model(int8_xml_path)
