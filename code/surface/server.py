#!/usr/bin/env python
# -*- coding=utf-8 -*-
#coding=utf-8
#接收图像并回传处理结果
import hiai
import socket
import threading
import time
import sys
import os
import struct
import cv2 as cv
import numpy as np
from PIL import Image
import json
import copy
import numpy
from hiai.nntensor_list import NNTensorList
from hiai.nn_tensor_lib import NNTensor

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'convert-6194-retina.om')
resource_path = os.path.join(current_dir, 'test_data')
output_path = os.path.join(current_dir, 'output')
anchors_npy = os.path.join(current_dir, "anchors_d310.npy")
if not os.path.exists(resource_path):
    print("no test data, please check. resource_path = %s" % resource_path)
    exit(0)

with open(os.path.join(current_dir, 'index'), 'r') as f:
  index_map = json.loads(f.read())
class_names = index_map['labels_list']
image_shape = index_map['image_shape']

net_h = int(image_shape[0])
net_w = int(image_shape[1])

class_num = len(class_names)

stride_list = [8, 16, 32]
anchors_1 = np.array([[10, 13], [16, 30], [33, 23]]) / stride_list[0]
anchors_2 = np.array([[30, 61], [62, 45], [59, 119]]) / stride_list[1]
anchors_3 = np.array([[116, 90], [156, 198], [163, 326]]) / stride_list[2]
anchor_list = [anchors_1, anchors_2, anchors_3]
###
net_h = 640
net_w = 640
conf_threshold = 0.3
iou_threshold = 0.45
_scale_factors = [10.0, 10.0, 5.0, 5.0]
channel_means = [123.68, 116.779, 103.939]
###
conf_threshold = 0.3
iou_threshold = 0.4

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]




def image_pre_process(image, aipp_flag=True):
    img_w, img_h = image.size
    image = image.convert('RGB')
    img = image.resize((net_w, net_h))
    img = np.array(img)
    if not aipp_flag:
        # Not using appi
        img = img - channel_means
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1)).copy()
    else:
        # Using appi
        img = img.astype(np.int8).copy()

    return img, img_w, img_h

def overlap(x1, x2, x3, x4):
  left = max(x1, x3)
  right = min(x2, x4)
  return right - left


def cal_iou(box, truth):
  w = overlap(box[0], box[2], truth[0], truth[2])
  h = overlap(box[1], box[3], truth[1], truth[3])
  if w <= 0 or h <= 0:
    return 0
  inter_area = w * h
  union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
  return inter_area * 1.0 / union_area


def apply_nms(all_boxes, thres):
  res = []

  for cls in range(class_num):
    cls_bboxes = all_boxes[cls]
    sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

    p = dict()
    for i in range(len(sorted_boxes)):
      if i in p:
        continue

      truth = sorted_boxes[i]
      for j in range(i + 1, len(sorted_boxes)):
        if j in p:
          continue
        box = sorted_boxes[j]
        iou = cal_iou(box, truth)
        if iou >= thres:
          p[j] = 1

    for i in range(len(sorted_boxes)):
      if i not in p:
        res.append(sorted_boxes[i])
  return res


def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
  def _sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

  _, h, w = conv_output.shape
  pred = conv_output.transpose((1, 2, 0)).reshape((h * w, 3, 5 + class_num))

  pred[..., 4:] = _sigmoid(pred[..., 4:])
  pred[..., 0] = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
  pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
  pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
  pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

  bbox = np.zeros((h * w, 3, 4))
  bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
  bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
  bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
  bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max

  pred[..., :4] = bbox
  pred = pred.reshape((-1, 5 + class_num))
  pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
  pred = pred[pred[:, 4] >= conf_threshold]
  pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)

  all_boxes = [[] for ix in range(class_num)]
  for ix in range(pred.shape[0]):
    box = [int(pred[ix, iy]) for iy in range(4)]
    box.append(int(pred[ix, 5]))
    box.append(pred[ix, 4])
    all_boxes[box[4] - 1].append(box)

  return all_boxes

'''
def get_result(model_outputs, img_w, img_h, new_w, new_h, shift_x_ratio, shift_y_ratio):
  num_channel = 3 * (class_num + 5)
  x_scale = net_w / float(new_w)
  y_scale = net_h / float(new_h)
  all_boxes = [[] for ix in range(class_num)]
  for ix in range(3):
    pred = model_outputs[2 - ix].reshape((num_channel, net_h // stride_list[ix], net_w // stride_list[ix]))
    anchors = anchor_list[ix]
    boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
    all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_num)]

  res = apply_nms(all_boxes, iou_threshold)

  return res
'''

def _sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def postprocess(anchors, box_encodings, class_predictions, ori_h, ori_w):
    detection_boxes = batch_decode(box_encodings, anchors, class_predictions, ori_h, ori_w)
    detection_scores_with_background = _sigmoid(class_predictions)
    detection_scores = detection_scores_with_background[:, :, 1:]
    detection_score = np.expand_dims(np.max(detection_scores[:, :, 0:], axis=-1), 2)
    detection_cls = np.expand_dims(np.argmax(detection_scores[:, :, 0:], axis=-1), 2)

    preds = np.concatenate([detection_boxes, detection_score, detection_cls], axis=2)
    res_list = []
    for pred in preds:
        pred = pred[pred[:, 4] >= conf_threshold]
        all_boxes = [[] for ix in range(class_num)]
        for ix in range(pred.shape[0]):
            box = [int(pred[ix, iy]) for iy in range(4)]
            box.append(int(pred[ix, 5]))
            box.append(pred[ix, 4])
            all_boxes[box[4] - 1].append(box)

        res = apply_nms(all_boxes, iou_threshold)
        res_list.append(res)
    return res_list
def get_center_coordinates_and_sizes(bbox):
    ymin, xmin, ymax, xmax = np.transpose(bbox)
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]
def decode_bbox_frcnn(conv_output, anchors, ori_h, ori_w):
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ty, tx, th, tw = conv_output.transpose()
    ty /= _scale_factors[0]
    tx /= _scale_factors[1]
    th /= _scale_factors[2]
    tw /= _scale_factors[3]

    w = np.exp(tw) * wa
    h = np.exp(th) * ha

    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a

    scale_h = 1
    scale_w = 1
    ymin = np.maximum((ycenter - h / 2.) * scale_h * ori_h, 0)
    xmin = np.maximum((xcenter - w / 2.) * scale_w * ori_w, 0)
    ymax = np.minimum((ycenter + h / 2.) * scale_h * ori_h, ori_h)
    xmax = np.minimum((xcenter + w / 2.) * scale_w * ori_w, ori_w)

    bbox_list = np.transpose(np.stack([ymin, xmin, ymax, xmax]))

    return bbox_list
def combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = list(tensor.shape)
  dynamic_tensor_shape = tensor.shape
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape
def batch_decode(box_encodings, anchors, class_predictions, ori_h, ori_w):
    combined_shape = combined_static_and_dynamic_shape(box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = np.tile(
        np.expand_dims(anchors, 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = np.reshape(tiled_anchor_boxes, [-1, 4])
    decoded_boxes = decode_bbox_frcnn(
        np.reshape(box_encodings, [-1, 4]),tiled_anchors_boxlist, ori_h, ori_w)
    decoded_boxes = np.reshape(decoded_boxes, np.stack(
        [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes
def get_result(result, img_w, img_h):
    anchors = np.load(anchors_npy)
    box_encodings = np.transpose(np.reshape(result[1], [4, -1, 1]))
    class_predictions_with_background = np.transpose(np.reshape(result[2], [result[2].shape[-1], -1, 1]))
    detection_dict = postprocess(anchors, box_encodings,
                                 class_predictions_with_background,
                                 img_h, img_w)

    return detection_dict


class Graph():
    def __init__(self, model_path):
        # self.graph = self.CreateGraph(model_path)
        self.model_path = model_path
        self.graph = self.CreateGraph()

    def CreateGraph(self):
        '''
        Create graph

        Returns:
            graph
        '''
        path1, filename =  os.path.split(self.model_path)
        nntensorlist_object =hiai.NNTensorList()
        # graphId = random.randint(1, 2**32-1)
        graph = hiai.Graph(hiai.GraphConfig(graph_id=65536))
        with graph.as_default():
            engine = hiai.Engine()
            #resize_config = hiai.ResizeConfig(resize_width=300, resize_height = 300)
            #nntensorlist_object = engine.resize(input_tensor_list=nntensorlist_object, config=resize_config)

            ai_model_desc = hiai.AIModelDescription(filename, self.model_path)
            if not os.path.exists(self.model_path):
                print(self.model_path)
                print("om model not existed.")
            ai_config = hiai.AIConfig(hiai.AIConfigItem("Inference", "item_value_2"))
            final_result = engine.inference(input_tensor_list=nntensorlist_object,
                                        ai_model=ai_model_desc,
                                        ai_config=ai_config)
        ret = copy.deepcopy(graph.create_graph())
        if ret != hiai.HiaiPythonStatust.HIAI_PYTHON_OK:
            graph.destroy()
            raise Exception("create graph failed, ret ", ret)
        print("create graph successful")
        return graph

    def __del__(self):
        '''
        Destroy graph
        '''
        self.graph.destroy()




class DemoService():
    def _preprocess(self, pic_path):
        self.input_width = int(image_shape[0])
        self.input_height = int(image_shape[1])
        self.aipp_flag = True
        # preprocessed_data = {}
        # images = []

        # preprocessed_data = []        


        input_rgb = Image.open(pic_path)
        img_preprocess, self.img_w, self.img_h = image_pre_process(input_rgb, self.aipp_flag)
        # 单batch下传入一个numpy数组
        tensor = NNTensor(img_preprocess)
        # images.append(tensor)
        # 构建一个输入
        tensor_list = NNTensorList([tensor])

            # # preprocessed_data是这样的[("pic_path",tensor_list), ("pic_path2",tensor_list)]
            # preprocessed_data.append((pic_path, tensor_list))

        # preprocessed_data['images'] = tensor_list

        return tensor_list
    
    def _postprocess(self, data):
        #for k, v in data.items():
            result = get_result(data, self.img_w, self.img_h)
            response = {
                'detection_classes': [],
                'detection_boxes': [],
                'detection_scores': []
            }
            for i in range(len(result[0])):
                ymin = result[0][i][0]
                xmin = result[0][i][1]
                ymax = result[0][i][2]
                xmax = result[0][i][3]
                score = result[0][i][5]
                label = result[0][i][4]

                if score < conf_threshold:
                    continue
                response['detection_classes'].append(
                    class_names[int(label)])
                response['detection_boxes'].append([
                    str(ymin), str(xmin),
                    str(ymax), str(xmax)
                ])
                response['detection_scores'].append(str(score))
            return response


    def convert_labels(self, label_list):
        """
            class_names = ['person', 'face']
            :param label_list: [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]
            :return: 
            """
        if isinstance(label_list, np.ndarray):
            label_list = label_list.tolist()
            label_names = [class_names[int(index)] for index in label_list]
        return label_names

    def ping(self):
        return

    def signature(self):
        pass


def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 23456))#这里换上自己的ip和端口
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print ("Waiting...")

    def recvall(sock, count):
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
    # 没有连接则等待有连接
    conn, addr = s.accept()
    print('Accept new connection from:' + str(addr))
    i = 0
    while 1:
        #start = time.time()  # 用于计算帧率信息
        length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
        if not length:
            break
        stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
        data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
        decimg = cv.imdecode(data, cv.IMREAD_COLOR)  # 将数组解码成图像
        fn = "test" + str(i) + ".jpg"
        cv.imwrite(fn, decimg)
        i = i+1
        if i == 10:
            i = 0
        print ("Receive Success")

        pic_path = "./"+fn
        input_data = demo._preprocess(pic_path)
        result = {}
        result = my_graph.graph.proc(input_data)
        result_return = demo._postprocess(result)
        print ("Analyse Success")
        #(file_path, file_name) = os.path.split(pic_path)
        #output_file = os.path.join(output_path, file_name)
        #print(result_return)
        json_string = json.dumps(result_return)
        conn.send(json_string.encode())
        print("Send Feedback Over")
    s.close()

def deal_data(conn, addr):
    print ('Accept new connection from {0}'.format(addr))
    while 1:
        fileinfo_size = struct.calcsize('128sl')
        buf = conn.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.strip(str.encode('\00'))
            new_filename = os.path.join(str.encode('./test_data/'), str.encode('atlas_') + fn)
            print ('file new name is {0}, filesize is {1}'.format(new_filename, filesize))

            recvd_size = 0  # 定义已接收文件的大小
            fp = open(new_filename, 'wb')
            print ("Start Receiving...")
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print ("Receive Success")
            pic_path = new_filename
            input_data = demo._preprocess(pic_path)
            result = {}
            result = my_graph.graph.proc(input_data)
            result_return = demo._postprocess(result)
            #(file_path, file_name) = os.path.split(pic_path)
            #output_file = os.path.join(output_path, file_name)
            #print(result_return)
            json_string = json.dumps(result_return)
            #msg = "Perry has received your file."
            conn.send(json_string.encode())
            print ("Analyse Success")
            print("Send Feedback Over")
        conn.close()
        break

if __name__ == '__main__':
    # 先创建图（流程编排）
    my_graph = Graph(model_path)

    if not os.path.exists(output_path):
      os.mkdir(output_path)

    src_dir = os.listdir(resource_path)

    demo = DemoService()

    socket_service()