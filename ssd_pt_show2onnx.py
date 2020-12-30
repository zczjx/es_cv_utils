#!/usr/bin/env python3
# coding: utf-8

import os, time, sys, pickle
import argparse
from net_gen_base import net_gen_base
sys.path.append("..")
from ssd import build_ssd
import torch
import onnx

class ssd_gen(net_gen_base):
    def __init__(self, cfg=None, width=300, height=300, num_classes=21):
        self.imgsz = width
        self.num_classes = num_classes

    def gen_net_model(self, pt_weight_file):
        net = build_ssd('test', self.imgsz, self.num_classes)            # initialize SSD
        net.load_state_dict(torch.load(pt_weight_file))
        net.eval()
        return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/ssd300_COCO_35000.pth', help='weights path')
    opt = parser.parse_args()

    net_gen = ssd_gen()
    ssd_net = net_gen.gen_net_model(opt.weights)

    # ssd_net.fuse()
    imgsz = (300, 300)
    img = torch.zeros((1, 3) + imgsz)  # (1, 3, 416, 416)
    f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
    torch.onnx.export(ssd_net, img, f, verbose=False, opset_version=11,
                    input_names=['images'], output_names=['detections'])

    model_onnx = onnx.load(f)  # Load the ONNX model
    onnx.checker.check_model(model_onnx)  # Check that the IR is well formed
    print(onnx.helper.printable_graph(model_onnx.graph))  # Print a human readable representation of the graph