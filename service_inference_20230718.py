import os
import cv2
import time
import math
import shutil
import random
import argparse
import requests
import subprocess
import numpy as np
import _init_paths
from math import *
from PIL import Image
import onnxruntime as ort
from service_keypoints import *
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

names = ['outer', 'inter', 'hole']

logger = modbus_tk.utils.create_logger("console")


# pose estimation

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def getPose_fromT(T):
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]

    R = T[:3, :3]
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0

    return x, y, z, rx, ry, rz


def getT_fromPose(x, y, z, rx, ry, rz):
    Rx = np.mat([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.mat([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.mat([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])

    t = np.mat([[x], [y], [z]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    R_ = np.array(R)
    t_ = np.array(t)
    T_1 = np.append(R_, t_, axis=1)

    zero = np.mat([0, 0, 0, 1])
    T_2 = np.array(zero)

    T = np.append(T_1, T_2, axis=0)
    T = np.mat(T)

    return T


# x, y, z, rx, ry, rz (rad)
# T_cam_outer = getT_fromPose(102.27699808600157, -129.187267392693, -124.89554283591471, -0.0070771838067699155,
#                             0.030311293903197367, 0.2124532269930333)
# T_cam_inter = getT_fromPose(6.331879509048326, -107.1656270372814, -89.14246268978107, -0.04350802136341767,
#                             0.03819525155739031, 0.1699569345297337)
# T_cam_hole = getT_fromPose(20.960830323172882, -239.55196534057822, -299.19566382196894, 0.004050616030267564,
#                            -0.04914603772928844, 0.09120093271427267)

T_cam_outer = getT_fromPose(92.35614037955287, -156.98579767576794, -126.02503822301642, 0.07611160456020122,
                            0.0004299451192411748, 0.08394145910064293)
T_cam_inter = getT_fromPose(0.6939026209463647, -109.78087497472619, -76.97487063484073, 0.005346074169449418,
                            0.02884951455362796, 0.0804989902238099)
T_cam_hole = getT_fromPose(1.54002411982046, -256.18371747890177, -283.1184815247451, -0.00631423145545828,
                           -0.04930301816394878, 0.009427174150475212)

# error
# v_error_outer = getT_fromPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# v_error_inter = getT_fromPose(24.66, -10.9, -17.7, 0.0, 0.0, 0.0)
# v_error_hole = getT_fromPose(-6.713, -4.86, -13.03, 0.0, 0.0, 0.0)
v_error_outer = getT_fromPose(-15.56, -3.69, 9.47, 0.0, 0.0, 0.0)
v_error_inter = getT_fromPose(-1.93, -1.91, -0.12, 0.0, 0.0, 0.0)
v_error_hole = getT_fromPose(-10.45, 0.0, -2.64, 0.0, 0.0, 0.0)

# 二次拍照修正位置
c_error_outer = getT_fromPose(28.871, 90.656, -56.658 + 90, -10 / 57.3, 0.0, 0.0)
c_error_inter = getT_fromPose(0.0, 0.0, 100.0, 0.0, 0.0, 0.0)
c_error_hole = getT_fromPose(147.725, -1.977, -136.36, 0.0, 0.0, 0.0)


class ONNX_engine():
    def __init__(self, weights, size, cuda) -> None:
        self.img_new_shape = (size, size)
        self.weights = weights
        self.device = cuda
        self.init_engine()
        self.names = names
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}

    def init_engine(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.weights, providers=providers)

    def predict(self, im):
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        outputs = self.session.run(outname, inp)[0]
        return outputs

    def preprocess(self, src):
        self.img = src
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        image = self.img.copy()
        im, ratio, dwdh = self.letterbox(image, auto=False)
        # t1 = time.time()
        outputs = self.predict(im)
        # print("inference time", (time.time() - t1) * 1000, ' ms')
        ori_images = [self.img.copy()]
        bbox = [[(0, 0), (0, 0)]]
        name = None
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = self.names[cls_id]
            color = self.colors[name]
            # name += ' ' + str(score)
            bbox = [[(box[0], box[1]), (box[2], box[3])]]
            # cv2.rectangle(image, box[:2], box[2:], color, 2)
            # cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=1)

        return ori_images[0], bbox, name, image

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]
        new_shape = self.img_new_shape

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255
        return im, r, (dw, dh)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov7', type=str, default='./tools/logistics/robot.onnx', help='weights path of onnx')
    parser.add_argument('--cfg', type=str, default='./tools/logistics/keypoints.yaml')

    parser.add_argument('--outer', type=str, default='./tools/logistics/outer.pth')
    parser.add_argument('--inter', type=str, default='./tools/logistics/inter.pth')
    parser.add_argument('--hole', type=str, default='./tools/logistics/hole.pth')

    parser.add_argument('--outer_3d', type=str, default='./estimation/01-model/outer_3d.json')
    parser.add_argument('--inter_3d', type=str, default='./estimation/01-model/inter_3d.json')
    parser.add_argument('--hole_3d', type=str, default='./estimation/01-model/hole_3d.json')
    parser.add_argument('--intrinsic', type=str, default='./estimation/02-intrinsic/ww_3x3.json')
    parser.add_argument('--save_root', type=str, default='./estimation/03-pred-pts2')
    parser.add_argument('--ret_root', type=str, default='./estimation/04-result')
    parser.add_argument('--solution', type=str, default='./estimation/estimation.exe')

    parser.add_argument('--cuda', type=bool, default=True, help='if your pc have cuda')
    parser.add_argument('--size', type=int, default=640, help='infer the img size')

    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    return args


# python .\tools\service_inference_20230718.py

if __name__ == '__main__':

    opt = parse_args()
    outer_model, inter_model, hole_model = init_keypoints(opt)
    onnx_engine = ONNX_engine(opt.yolov7, opt.size, opt.cuda)

    outer_3d = opt.outer_3d
    inter_3d = opt.inter_3d
    hole_3d = opt.hole_3d
    intrinsic = opt.intrinsic
    save_root = opt.save_root
    ret_root = opt.ret_root
    solution = opt.solution

    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    if os.path.exists(ret_root):
        shutil.rmtree(ret_root)

    os.makedirs(save_root)
    os.makedirs(ret_root)

    try:
        master = modbus_tcp.TcpMaster(host="192.168.2.150")  # 车库ip
        master.set_timeout(5.0)
        logger.info("connected")

        while True:
            tags = master.execute(1, cst.READ_HOLDING_REGISTERS, 1000, 1)
            logger.info(tags[0])

            # tag == 1 打开摄像头开始采集：
            if tags[0] == 1:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
                cap.set(cv2.CAP_PROP_FPS, 60)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
                flag = cap.isOpened()

                target_model = None
                pose_model = None
                error_tag = None
                class_name = None
                v_error = None
                c_error = None
                T_center = None
                index = 0

                while flag and tags[0] == 1:
                    src, frame = cap.read()
                    image = frame
                    img_show = frame
                    start1 = time.time()
                    # height, width, channels = image.shape[:3]
                    detect_img, box, class_name, img_det = onnx_engine.preprocess(image)
                    # img_roi = get_img_roi(image, box, height, width)

                    if class_name == 'outer':
                        target_model = outer_3d
                        pose_model = outer_model
                        v_error = v_error_outer
                        c_error = c_error_outer

                    elif class_name == 'inter':
                        target_model = inter_3d
                        pose_model = inter_model
                        v_error = v_error_inter
                        c_error = c_error_inter

                    elif class_name == 'hole':
                        target_model = hole_3d
                        pose_model = hole_model
                        v_error = v_error_hole
                        c_error = c_error_hole

                    else:
                        print('Cannot Find The Target\n ')
                        class_name = 'None'

                    if not class_name == 'None':
                        data_preds, img_bgr = pose_landmark(box, pose_model, image, image)
                        img_show = img_bgr
                        if data_preds.all() != np.zeros([8, 3], dtype=float).all():
                            save_path = save_root + '/image_' + str(index) + '.json'
                            ret_path = ret_root + '/image_' + str(index) + '.json'
                            np.savetxt(save_path, data_preds, fmt='%0.8f', delimiter='\t')
                            subprocess.run([solution, target_model, intrinsic, save_path, ret_path])
                            index = index + 1

                            if os.path.exists(ret_path):
                                # 读取数据文件写入寄存器
                                with open(ret_path, 'r') as f:
                                    data = f.read()
                                    if len(data) != 0:
                                        # 从文件中读取初始姿态参数
                                        data_split = data.split()
                                        x, y, z, rx, ry, rz = float(data_split[3]), float(data_split[4]), float(
                                            data_split[5]), float(data_split[0]), float(data_split[1]), float(
                                            data_split[2])
                                        T_tag_cam = getT_fromPose(x, y, z, rx, ry, rz)
                                        print('T_tag_cam\n', np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2)))

                                        # 读取机械臂实时位姿信息
                                        arm_pose = master.execute(1, cst.READ_HOLDING_REGISTERS, 1030, 6)
                                        arm_pose_list = list(arm_pose)
                                        # 置为负值
                                        i = 0
                                        while i < len(arm_pose_list):
                                            if arm_pose_list[i] > 32767:
                                                arm_pose_list[i] = arm_pose_list[i] - 65536
                                            i += 1

                                        arm_x = arm_pose_list[0] / 10
                                        arm_y = arm_pose_list[1] / 10
                                        arm_z = arm_pose_list[2] / 10
                                        arm_rx = arm_pose_list[3] / 1000
                                        arm_ry = arm_pose_list[4] / 1000
                                        arm_rz = arm_pose_list[5] / 1000
                                        print('机械臂姿态参数: ', arm_x, arm_y, arm_z, arm_rx * 57.3, arm_ry * 57.3,
                                              arm_rz * 57.3)

                                        T_tools_base = getT_fromPose(arm_x, arm_y, arm_z, arm_rx, arm_ry, arm_rz)
                                        T_tag_mechanism = np.zeros([4, 4], dtype=float)

                                        # 设置坐标转换关系
                                        if class_name == 'outer':
                                            T_tag_mechanism = v_error * T_tools_base * T_cam_outer * T_tag_cam

                                        elif class_name == 'inter':
                                            T_tag_mechanism = v_error * T_tools_base * T_cam_inter * T_tag_cam

                                        elif class_name == 'hole':
                                            T_tag_mechanism = v_error * T_tools_base * T_cam_hole * T_tag_cam

                                        tag_m_x, tag_m_y, tag_m_z, tag_m_rx, tag_m_ry, tag_m_rz = getPose_fromT(
                                            T_tag_mechanism)
                                        tag_m_x = tag_m_x
                                        tag_m_y = tag_m_y
                                        tag_m_z = tag_m_z
                                        tag_m_rx = tag_m_rx
                                        tag_m_ry = tag_m_ry
                                        tag_m_rz = tag_m_rz

                                        print('T_tag_mechanism: ', tag_m_x, tag_m_y, tag_m_z, tag_m_rx * 57.3,
                                              tag_m_ry * 57.3, tag_m_rz * 57.3)

                                        # 转换进制,变成整数发送给PLC
                                        # mm * 10
                                        conv_x = int(tag_m_x * 10)
                                        conv_y = int(tag_m_y * 10)
                                        conv_z = int(tag_m_z * 10)

                                        conv_rx = int(tag_m_rx * 1000)
                                        conv_ry = int(tag_m_ry * 1000)
                                        conv_rz = int(tag_m_rz * 1000)

                                        # 计算拍照位
                                        T_center = c_error * T_tag_mechanism
                                        camera_x, camera_y, camera_z, camera_rx, camera_ry, camera_rz = getPose_fromT(
                                            T_center)
                                        print('T_center: \t', camera_x, camera_y, camera_z, camera_rx * 57.3,
                                              camera_ry * 57.3, camera_rz * 57.3)
                                        # 相机拍照位传输单位换算
                                        conv_camera_x = int(camera_x * 10)
                                        conv_camera_y = int(camera_y * 10)
                                        conv_camera_z = int(camera_z * 10)

                                        conv_camera_rx = int(camera_rx * 1000)
                                        conv_camera_ry = int(camera_ry * 1000)
                                        conv_camera_rz = int(camera_rz * 1000)

                                        if x != 0 and y != 0 and z != 0:
                                            # 位姿信息写入寄存器
                                            master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 1010,
                                                           output_value=[conv_x, conv_y, conv_z, conv_rx, conv_ry,
                                                                         conv_rz])
                                            master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 1020,
                                                           output_value=[conv_camera_x, conv_camera_y, conv_camera_z,
                                                                         conv_camera_rx, conv_camera_ry,
                                                                         conv_camera_rz])

                                            # 将1007位设置为目标类别的标识位
                                            if class_name == 'outer':
                                                master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 1007, output_value=[1])
                                            elif class_name == 'inter':
                                                master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 1007, output_value=[2])
                                                master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 1008,
                                                               output_value=[595])
                                            elif class_name == 'hole':
                                                master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 1007, output_value=[3])

                                            end1 = time.time()
                                            print("cost time: ", end1 - start1, " s")

                                            print("PLC内部的 x, y, z , rx, ry, rz 为：",
                                                  master.execute(1, cst.READ_HOLDING_REGISTERS, 1010, 6))
                                            print("-----------------------------------")
                                    else:
                                        print("file is empty!")
                            else:
                                print("cannot find the json file!")

                    cv2.imshow('imshow', img_show)
                    k = cv2.waitKey(1)
                    # 第几位， 读几个
                    tags1 = master.execute(1, cst.READ_HOLDING_REGISTERS, 1000, 1)
                    if tags1[0] == 0 or k == 27:
                        cv2.destroyAllWindows()
                        break
            # 如果tag == 0, 关闭摄拍照
            elif tags[0] == 0:
                time.sleep(1.0)
                print("The camera is off!")
    except modbus_tk.modbus.ModbusError as e:
        logger.error("%s- Code=%d" % (e, e.get_exception_code()))
