import argparse
import time
from pathlib import Path
import kserve
import numpy as np
import os
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import json
import base64
from PIL import Image
import io

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class Model(kserve.Model):
    def __init__(self, name, model_path, device, args):
        super().__init__(name)
        self.path = model_path
        # 确保 device 参数是正确的格式
        self.device = self._parse_device(device)
        self.args = args
        self.model = None
        self.names = None
        self.stride = None
        self.ready = False

    def _parse_device(self, device_str):
        """解析设备参数，确保格式正确"""
        if device_str.lower() == 'cpu':
            return 'cpu'
        elif device_str.lower() == 'cuda':
            # 如果没有指定设备 ID，默认使用 0
            return '0' if torch.cuda.is_available() else 'cpu'
        else:
            # 检查是否是有效的设备 ID 格式，如 '0', '0,1' 等
            try:
                # 验证设备 ID 是否存在
                if torch.cuda.is_available():
                    ids = [int(d) for d in device_str.split(',')]
                    for i in ids:
                        if i >= torch.cuda.device_count():
                            print(f"Warning: Device ID {i} is invalid, using CPU instead")
                            return 'cpu'
                return device_str
            except ValueError:
                print(f"Invalid device format: {device_str}, using CPU instead")
                return 'cpu'

    def load(self):
        try:
            # 确保 device 参数是正确的格式
            self.device = self._parse_device(self.device)
            
            # 打印设备信息进行调试
            print(f"Selected device ID: {self.device}")
            
            # Initialize device using the correct format
            self.device = select_device(self.device)

            # Load model
            print(f"Loading model from {self.path} to device {self.device}")
            self.model = attempt_load(self.path, map_location=self.device)  # load FP32 model
            if self.model is None:
                print("Failed to load model.")
                return False
            
            self.stride = int(self.model.stride.max())  # model stride
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

            # Half precision
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            if self.half:
                self.model.half()  # to FP16

            # Warmup
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.args.img_size, self.args.img_size).to(self.device).type_as(next(self.model.parameters())))

            print(f"Model loaded on {self.device.type}")
            self.ready = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.ready = False
            return False

    def predict(self, payload, scale_factor=1, out_threshold=0.5):
        if not self.ready:
            raise RuntimeError("Model is not ready")
            
        if isinstance(payload, bytes):
            payload = json.loads(payload)

        inputs = payload["instances"]
        data = inputs[0]["image"]
        raw_img_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(raw_img_data)).convert('RGB')
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert args to proper format
        class Args:
            def __init__(self):
                pass

        opt = Args()
        opt.source = 'input_image'
        opt.weights = self.path
        opt.img_size = self.args.img_size
        opt.conf_thres = self.args.conf_thres
        opt.iou_thres = self.args.iou_thres
        opt.device = self.device  # Use the already initialized device
        opt.view_img = False
        opt.save_txt = False
        opt.save_conf = False
        opt.save_crop = False
        opt.nosave = True
        opt.classes = None
        opt.agnostic_nms = False
        opt.augment = False
        opt.update = False
        opt.project = ''
        opt.name = ''
        opt.exist_ok = True
        opt.line_thickness = 3
        opt.hide_labels = False
        opt.hide_conf = False

        # Run detection
        results = self.detect(opt, img)

        # Convert result image to base64
        _, buffer = cv2.imencode('.png', results['image'])
        img_str = base64.b64encode(buffer).decode('utf-8')

        return {"predictions": img_str}

    def detect(self, opt, input_img=None):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Prepare input
        if input_img is not None:
            # Use provided image
            im0s = input_img.copy()
            img = letterbox(im0s, imgsz, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)
            dataset = [(source, img, im0s, None)]
        else:
            # Set Dataloader
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True
                dataset = LoadStreams(source, img_size=imgsz, stride=self.stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=self.stride)

        results = {'image': None, 'detections': []}
        t0 = time.time()

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Store detections and draw boxes
                    for *xyxy, conf, cls in reversed(det):
                        # Store detection info
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        results['detections'].append({
                            'class': self.names[int(cls)],
                            'confidence': float(conf),
                            'bbox': xywh
                        })

                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (self.names[c] if opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Store the processed image
                results['image'] = im0

        print(f'Done. ({time.time() - t0:.3f}s)')
        return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='default', help='训练好的模型权重文件路径 (.pth)')
    # 恢复 --devices 参数，但将其映射到 --device
    parser.add_argument('--devices', type=str, default='', help='运行设备 (cuda 或 cpu)') 
    parser.add_argument('--device', type=str, default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--save_dir', default='default_save', help='save_dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_name
    
    # 处理 --devices 和 --device 参数
    if args.devices:
        # 如果同时指定了 --devices 和 --device，优先使用 --devices
        if args.device and args.devices != args.device:
            print(f"Warning: Both --devices and --device are specified. Using --devices={args.devices}")
        args.device = args.devices
    
    # 设置默认设备
    if not args.device:
        args.device = '0' if torch.cuda.is_available() else 'cpu'
    
    device = args.device
    
    # 打印一些调试信息
    print(f"Args: {args}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        sys.exit(1)

    model = Model("custom-model", model_path, device, args)
    
    # 显式调用 load() 并检查返回值
    if model.load():
        print("Model loaded successfully and ready")
        # 验证 ready 属性
        print(f"Model ready status: {model.ready}")
        kserve.ModelServer(workers=1).start([model])
    else:
        print("Model failed to load. Exiting.")
        sys.exit(1)