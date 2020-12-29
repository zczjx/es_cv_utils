#!/usr/bin/env python3
# coding: utf-8

import os, time, sys, pickle
import netron


if __name__ == '__main__':
    netron.start(sys.argv[1], port=int(sys.argv[2]), host='192.168.2.110')