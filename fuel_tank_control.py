import argparse
import time
from pathlib import Path

import cv2
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, strip_optimizer, \
    set_logging, increment_path
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized


def solveEPnP(pred_points):
    # 3D model points.
    model_points = np.array([(-343, 540, 0.0),
                             (-343, -540, 0.0),
                             (343, -540, 0.0),
                             (343, 540, 0.0),
                             (-343, 540, 660),
                             (-343, -540, 660),
                             (343, -540, 660),
                             (343, 540, 660)])

    # 设置相机参数矩阵
    camera_matrix = np.array(
        [[746.350022499676, 0, 307.115400875528],
         [0, 994.741072914622, 247.859498440559],
         [0, 0, 1]], dtype=np.double)

    dist_coeffs = np.array(
        (-0.434643650640820, 0.400527856579466, -0.431891889112567, 0, 0))  # Assuming no lens distortion
    # dist_coeffs = np.array((0, 0, 0, 0, 0))  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, pred_points, camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_EPNP)

    return rotation_vector, translation_vector


def detect(opt):
    global dataset
    source, weights, view_img, save_txt, imgsz, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.kpt_label

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir 20240109

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list, tuple)):
        assert len(imgsz) == 2;
        "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    cudnn.benchmark = True  # set True to speed up constant image size inference
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    while True:

        # 确定是否打开相机
        tags = master.execute(1, cst.READ_HOLDING_REGISTERS, 500, 1)
        print(tags[0])
        tag = tags[0]

        if tag == 1:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                # pred = model(img, augment=opt.augment)[0]
                pred = model(img)[0]
                # print(pred[..., 4].max())
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms,
                                           kpt_label=kpt_label, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'])

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    im0, frame = im0s[i].copy(), dataset.count

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                        scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class

                        # Write results
                        pred_points = np.zeros((8, 2), dtype=float)
                        for det_index, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                            # print('size',im0.shape[0])
                            '''-------转换keypoint--------------'''
                            keypoints = det[det_index, 6:]
                            keypoints = keypoints.tolist()  # tensor转list

                            index = 3
                            for kip in range(8):
                                x_pt0, y_pt0 = keypoints[index * kip], keypoints[index * kip + 1]

                                pred_points[kip][0] = x_pt0
                                pred_points[kip][1] = y_pt0

                            '''--------------EPnP位姿估计，并输出结果--------------'''
                            rotation_vector, translation_vector = solveEPnP(pred_points)
                            rx = rotation_vector[0][0] * 57.3
                            ry = rotation_vector[1][0] * 57.3
                            rz = rotation_vector[2][0] * 57.3
                            # 输出结果
                            print('-------------------------------------------')
                            angle = ry
                            distance = translation_vector[2][0]
                            print('相机测算角度: ', round(angle, 3), '°')
                            print('相机测算距离: ', round(distance / 1000, 3), 'm')
                            # 输出写入PLC的结果
                            plc_angle = int(angle * 10)
                            plc_distance = int(distance)
                            print('写入PLC的角度: ', plc_angle, '°*10')
                            print('写入PLC的距离: ', plc_distance, 'mm')
                            # 将角度/°、距离/mm信息写入寄存器指定位置
                            master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 501, output_value=[plc_angle])
                            master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 502, output_value=[plc_distance])
                            master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 503, output_value=[1])

                            c = int(cls)  # integer class
                            label = None if opt.hide_labels else (
                                names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            kpts = det[det_index, 6:]
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                         line_thickness=opt.line_thickness,
                                         kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])

                        '''----存图-----'''
                        vid_img_name = '%05d' % frame + '.jpg'
                        save_path3 = str(save_dir / vid_img_name)
                        cv2.imwrite(save_path3, im0)  # 存检测图
                    else:
                        master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 503, output_value=[0])
                        print('未检测到目标')

                    # Print time (inference + NMS)
                    t2 = time_synchronized()
                    print(f'Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    cv2.imshow('detect', im0)
                    cv2.waitKey(1)  # 1 millisecond
                tags = master.execute(1, cst.READ_HOLDING_REGISTERS, 500, 1)
                if tags[0] == 0:
                    print('关闭摄像头')
                    dataset.cap.release()
                    cv2.destroyAllWindows()
                    break
        elif tag == 0:
            print('未开启摄像头')

            time.sleep(2)

            # Save results (image with detections)

    # print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/20240114.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.80, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
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
    parser.add_argument('--kpt-label', default=True, help='use keypoint labels')

    opt = parser.parse_args()

    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    try:
        # master = modbus_tcp.TcpMaster(host="192.168.2.10")  # PLC ip
        master = modbus_tcp.TcpMaster(host="127.0.0.1")  # 车库ip
        master.set_timeout(5.0)
        print("modbus connected")

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect(opt=opt)
                    strip_optimizer(opt.weights)
            else:
                detect(opt=opt)
    except modbus_tk.modbus.ModbusError as e:
        print("%s- Code=%d" % (e, e.get_exception_code()))
