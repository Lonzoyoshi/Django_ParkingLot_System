# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
import csv
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result, allFilePath, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge

# 定义颜色
clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
danger = ['危', '险']


# 四个点按照左上、右上、右下、左下排列
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视变换得到车牌小图
def four_point_transform(image, pts):
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# 加载检测模型
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


# 返回到原图坐标
def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords


# 获取车牌坐标以及四个角点坐标并获取车牌号
def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model, is_color=False):
    h, w, c = img.shape
    result_dict = {}
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness

    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    height = y2 - y1
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    class_label = int(class_num)  # 车牌的类型0代表单牌，1代表双层车牌
    roi_img = four_point_transform(img, landmarks_np)  # 透视变换得到车牌小图
    if class_label:  # 判断是否是双层车牌，是双牌的话进行分割后然后拼接
        roi_img = get_split_merge(roi_img)
    if not is_color:
        plate_number, rec_prob = get_plate_result(roi_img, device, plate_rec_model, is_color=is_color)  # 对车牌小图进行识别
    else:
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model,
                                                                           is_color=is_color)
    result_dict['rect'] = rect  # 车牌roi区域
    result_dict['detect_conf'] = conf  # 检测区域得分
    result_dict['landmarks'] = landmarks_np.tolist()  # 车牌角点坐标
    result_dict['plate_no'] = plate_number  # 车牌号
    result_dict['rec_conf'] = rec_prob  # 每个字符的概率
    result_dict['roi_height'] = roi_img.shape[0]  # 车牌高度
    result_dict['plate_color'] = ""
    if is_color:
        result_dict['plate_color'] = plate_color  # 车牌颜色
        result_dict['color_conf'] = color_conf  # 颜色得分
    result_dict['plate_type'] = class_label  # 单双层 0单层 1双层
    return result_dict


def write_plate_result(plate_no, plate_color, save_path):
    """将车牌信息增量写入txt文件"""
    result_txt = os.path.join(save_path, 'plate_results.txt')

    if os.path.exists(result_txt) and os.path.getsize(result_txt) > 1024 * 1024:
        open(result_txt, 'w').close()

    with open(result_txt, 'a', encoding='utf-8') as f:
        f.write(f"{plate_no},{plate_color}\n")
        f.flush()


def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size, is_color=False):
    conf_thres = 0.3  # 得分阈值
    iou_thres = 0.5  # nms的iou值
    dict_list = []
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device, plate_rec_model,
                                                     is_color=is_color)
                if result_dict['plate_no']:  # 每次识别到车牌都写入文件
                    plate_no = result_dict['plate_no']
                    plate_color = result_dict.get('plate_color', '')
                    write_plate_result(plate_no, plate_color, save_path)  # 调用写入函数
                dict_list.append(result_dict)
    return dict_list


# 车牌结果画出来
def draw_result(orgimg, dict_list, is_color=False):
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        landmarks = result['landmarks']
        result_p = result['plate_no']
        if result['plate_type'] == 0:  # 单层
            result_p += " " + result['plate_color']
        else:  # 双层
            result_p += " " + result['plate_color'] + "双层"
        result_str += result_p + " "
        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)

        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if rect_area[0] + labelSize[0][0] > orgimg.shape[1]:
            rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
        orgimg = cv2.rectangle(orgimg, (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))),
                               (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]),
                               (255, 255, 255), cv2.FILLED)
        orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1])),
                               (0, 0, 0), 21)
    print(result_str)
    return orgimg


def write_plate_result(plate_no, plate_color, save_path):
    """将车牌信息写入txt文件"""
    result_txt = os.path.join(save_path, 'plate_results.txt')

    # 确保目录存在
    os.makedirs(os.path.dirname(result_txt), exist_ok=True)

    with open(result_txt, 'a', encoding='utf-8') as f:
        f.write(f"{plate_no},{plate_color}\n")


def save_detection_results(img_path, dict_list, save_path):
    """保存检测结果到CSV文件"""
    results_file = os.path.join(save_path, 'detection_results.csv')

    # 检查文件是否存在,如果不存在则创建并写入表头
    file_exists = os.path.exists(results_file)

    with open(results_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['image_path', 'plate_number', 'plate_color'])

        # 写入检测结果
        for result in dict_list:
            writer.writerow([
                img_path,
                result.get('plate_no', ''),
                result.get('plate_color', '')
            ])


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt',
                        help='model.pt path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth',
                        help='model.pt path(s)')  # 车牌识别+颜色识别模型
    parser.add_argument('--is_color', type=bool, default=True, help='plate color')  # 是否识别颜色
    parser.add_argument('--image_path', type=str, help='source')  # 图片路径
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  # 网络输入图片大小
    parser.add_argument('--output', type=str, default='result', help='source')  # 图片结果保存的位置
    parser.add_argument('--video', type=str, default='', help='source')  # 视频的路径
    parser.add_argument('--input_type', type=str, choices=['image', 'video', 'camera'], help='输入类型')
    parser.add_argument('--input_path', type=str, help='输入路径')
    parser.add_argument('--camera_id', type=int, default=0, help='camera device id')
    opt = parser.parse_args()
    print(opt)

    # 确保输出路径存在
    save_path = opt.output
    # 只用传入的 save_path，不再拼接 static
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = load_model(opt.detect_model, device)
    plate_rec_model = init_model(device, opt.rec_model, is_color=opt.is_color)

    count = 0
    time_all = 0
    time_begin = time.time()

    if opt.input_type == 'camera':
        # 确保输出目录存在
        os.makedirs(save_path, exist_ok=True)

        # 写入 done.txt
        done_file_path = os.path.join(save_path, 'done.txt')
        with open(done_file_path, 'w') as f:
            f.write('Camera ready')

        cap = cv2.VideoCapture(opt.camera_id)  # 使用指定的摄像头ID
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            dict_list = detect_Recognition_plate(detect_model, frame, device,
                                                 plate_rec_model, opt.img_size, is_color=opt.is_color)
            detected_frame = draw_result(frame, dict_list)

            # 确保 temp.png 写入成功
            temp_image_path = os.path.join(save_path, 'temp.png')
            success = cv2.imwrite(temp_image_path, detected_frame)
            if not success:
                print(f"Error writing temp.png to {temp_image_path}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"实时检测已结束")
    elif opt.input_type == 'image':
        image_path = os.path.abspath(opt.image_path)  # 确保路径是绝对路径
        if not os.path.exists(image_path):
            print(f"路径不存在：{image_path}")
            exit(1)

        if os.path.isfile(image_path):  # 单个图片
            print(count, image_path, end=" ")
            img = cv_imread(image_path)
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                 is_color=opt.is_color)
            # save_detection_results(image_path, dict_list, save_path)  # 保存检测结果
            ori_img = draw_result(img, dict_list)
            img_name = os.path.basename(image_path)
            save_img_path = os.path.join(save_path, 'detected_' + img_name)  # 添加前缀 'detected_'
            cv2.imwrite(save_img_path, ori_img)
        else:  # 目录
            file_list = []
            allFilePath(image_path, file_list)
            for img_path in file_list:  # 遍历图片文件
                print(count, img_path, end=" ")
                img = cv_imread(img_path)
                if img is None:
                    continue
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                     is_color=opt.is_color)
                # save_detection_results(img_path, dict_list, save_path)  # 保存检测结果
                ori_img = draw_result(img, dict_list)
                img_name = os.path.basename(img_path)
                save_img_path = os.path.join(save_path, 'detected_' + img_name)  # 添加前缀 'detected_'
                cv2.imwrite(save_img_path, ori_img)
                count += 1
            print(f"sumTime time is {time.time() - time_begin} s, average pic time is {time_all / count}")
    elif opt.input_type == 'video':
        video_name = os.path.abspath(opt.input_path)  # 确保路径是绝对路径
        if not os.path.exists(video_name):
            print(f"视频文件不存在：{video_name}")
            exit(1)

        capture = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = capture.get(cv2.CAP_PROP_FPS)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(os.path.join(save_path, 'detected_result.mp4'), fourcc, fps, (width, height))
        frame_count = 0
        fps_all = 0

        while capture.isOpened():
            t1 = cv2.getTickCount()
            frame_count += 1
            print(f"第{frame_count} 帧", end=" ")
            ret, img = capture.read()
            if not ret:
                break
            img0 = copy.deepcopy(img)
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                 is_color=opt.is_color)
            ori_img = draw_result(img, dict_list)
            t2 = cv2.getTickCount()
            infer_time = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / infer_time
            fps_all += fps
            str_fps = f'fps:{fps:.4f}'
            cv2.putText(ori_img, str_fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(ori_img)
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"all frame is {frame_count}, average fps is {fps_all / frame_count} fps")

    # 非摄像头模式下，最后写 done.txt
    if opt.input_type != 'camera':
        done_file_path = os.path.join(save_path, 'done.txt')
        with open(done_file_path, 'w') as f:
            f.write('Processing completed')
