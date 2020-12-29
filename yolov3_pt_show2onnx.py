#!/usr/bin/env python3
# coding: utf-8

import os, time, sys, pickle
import argparse
from net_gen_base import net_gen_base
from ..models import *
import torch
import onnx

class yolov3_gen(net_gen_base):
    def __init__(self, cfg=None, width=416, height=461, num_classes=80):
        self.cfg = cfg
        self.imgsz=(width, height)

    def gen_net_model(pt_weight_file):
        net = Darknet(self.cfg, self.imgsz)
        net.load_state_dict(torch.load(pt_weight_file)['model'])
        net.eval()
        return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    opt = parser.parse_args()

    net_gen = yolov3_gen(opt.cfg)
    yolov3_net = net_gen.gen_net_model(opt.weights)

    yolov3_net.fuse()
    imgsz = (416, 416)
    img = torch.zeros((1, 3) + imgsz)  # (1, 3, 416, 416)
    f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
    torch.onnx.export(yolov3_net, img, f, verbose=False, opset_version=11,
                    input_names=['images'], output_names=['classes_conf', 'bbox'])

    model_onnx = onnx.load(f)  # Load the ONNX model
    onnx.checker.check_model(model_onnx)  # Check that the IR is well formed
    print(onnx.helper.printable_graph(model_onnx.graph))  # Print a human readable representation of the graph