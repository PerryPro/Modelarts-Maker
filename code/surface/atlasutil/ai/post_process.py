import numpy as np
import copy
from atlasutil.presenteragent.presenter_types import *
import cv2 as cv


def SSDPostProcess(inference_result, image_resolution, confidence_threshold, labels = []):
    '''
    processes SSD detection result, returns detection result

    Args:
        resultList: list, detection result
        image_resolution: integer, the quantity of categories with top confidece level user wants to obtain
        confidence_threshold: numpy array, the corresponding index of top n confidence
        labels: list of all categories that can be detected
    
    Returns:
        detection_result_list: list of ObjectDetectionResult
        detection_item.lt: left top coordinate with element x and y
        detection_item.rb: right below coordinate with element x and y
        detection_item.attr: label
        detection_item.confidence: confidence
    '''
    result = inference_result[0]
    shape = result.shape
    detection_result_list = []
    for i in range(0, shape[0]):
        item = result[i, 0, 0, ]
        if item[2] < confidence_threshold:
            continue
        detection_item = ObjectDetectionResult()
        detection_item.attr = int(item[1])
        detection_item.confidence = item[2]
        detection_item.lt.x = int(max(min(item[3], 1), 0) * image_resolution[1])
        detection_item.lt.y = int(max(min(item[4], 1), 0) * image_resolution[0])
        detection_item.rb.x = int(max(min(item[5], 1), 0) * image_resolution[1])
        detection_item.rb.y = int(max(min(item[6], 1), 0) * image_resolution[0])
        if labels == []:
            detection_item.result_text = str(detection_item.attr) + " " + str(round(detection_item.confidence*100,2)) + "%"
        else:
            detection_item.result_text = str(labels[detection_item.attr]) + " " + str(round(detection_item.confidence*100,2)) + "%"
        detection_result_list.append(detection_item)
    return detection_result_list 

anchors_yolo = [[(116,90),(156,198),(373,326)],[(30,61),(62,45),(59,119)],[(10,13),(16,30),(33,23)]]

def sigmoid(x):
    s = 1 / (1 + np.exp(-1*x))
    return s

#获取分数最高的类别,返回分数和索引
def getMaxClassScore(class_scores):
    class_score = 0
    class_index = 0
    for i in range(len(class_scores)):
        if class_scores[i] > class_score:
            class_index = i+1
            class_score = class_scores[i]
    return class_score,class_index

def getBBox(feat, anchors, image_shape, confidence_threshold):
    box = []
    for i in range(len(anchors)):
        for cx in range(feat.shape[0]):
            for cy in range(feat.shape[1]):
                tx = feat[cx][cy][0 + 85 * i]
                ty = feat[cx][cy][1 + 85 * i]
                tw = feat[cx][cy][2 + 85 * i]
                th = feat[cx][cy][3 + 85 * i]
                cf = feat[cx][cy][4 + 85 * i]
                cp = feat[cx][cy][5 + 85 * i:85 + 85 * i]

                bx = (sigmoid(tx) + cx)/feat.shape[0]
                by = (sigmoid(ty) + cy)/feat.shape[1]
                bw = anchors[i][0]*np.exp(tw)/image_shape[0]
                bh = anchors[i][1]*np.exp(th)/image_shape[1]

                b_confidence = sigmoid(cf)
                b_class_prob = sigmoid(cp)
                b_scores = b_confidence*b_class_prob
                b_class_score,b_class_index = getMaxClassScore(b_scores)

                if b_class_score > confidence_threshold:
                    box.append([bx,by,bw,bh,b_class_score,b_class_index])
    return box

#非极大值抑制阈值筛选得到bbox
def donms(boxes,nms_threshold):
    b_x = boxes[:, 0]
    b_y = boxes[:, 1]
    b_w = boxes[:, 2]
    b_h = boxes[:, 3]
    scores = boxes[:,4]
    areas = (b_w+1)*(b_h+1)
    order = scores.argsort()[::-1]
    keep = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(b_x[i], b_x[order[1:]])
        yy1 = np.maximum(b_y[i], b_y[order[1:]])
        xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
        yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
        #相交面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #相并面积,面积1+面积2-相交面积
        union = areas[i] + areas[order[1:]] - inter
        # 计算IoU：交 /（面积1+面积2-交）
        IoU = inter / union
        # 保留IoU小于阈值的box
        inds = np.where(IoU <= nms_threshold)[0]
        order = order[inds + 1]  # 因为IoU数组的长度比order数组少一个,所以这里要将所有下标后移一位

    final_boxes = [boxes[i] for i in keep]
    return final_boxes

def getBoxes(resultList, anchors, img_shape, confidence_threshold, nms_threshold):
    boxes = []
    for i in range(resultList):
        feature_map = resultList[i][0].transpose((2, 1, 0))
        box = getBBox(feature_map, anchors[i], img_shape, confidence_threshold)
        boxes.extend(box)
    Boxes = donms(np.array(boxes),nms_threshold)
    return Boxes

def Yolov3_post_process(resultList, confidence_threshold, nms_threshold, model_shape, img_shape, labels=[], anchors=anchors_yolo):
    '''
    processes YOLOv3 inference result, and returns detection result

    Args:
        resultList: list of inference result
        confidence_threshold: float number, confidence threshold
        nms_threshold: float number, NMS threshold
        model_shape: shape of model input
        img_shape: shape of original image
        labels: labels of model detection
        anchors: anchors of yolov3 model

    Returns:
        detection_result_list: list of ObjectDetectionResult
        detection_item.lt: left top coordinate with element x and y
        detection_item.rb: right below coordinate with element x and y
        detection_item.attr: label
        detection_item.confidence: confidence
    '''
    boxes = getBoxes(resultList, anchors, model_shape, confidence_threshold, nms_threshold)
    detection_result_list = []
    for box in boxes:
        detection_item = ObjectDetectionResult()
        if labels != []:
            detection_item.attr = labels[int(box[5])]
        else:
            detection_item.attr = ""
        detection_item.confidence = round(box[4],4)
        detection_item.lt.x = int((box[0]-box[2]/2)*img_shape[1])
        detection_item.lt.y = int((box[1]-box[3]/2)*img_shape[0])
        detection_item.rb.x = int((box[0]+box[2]/2)*img_shape[1])
        detection_item.rb.y = int((box[1]+box[3]/2)*img_shape[0])
        detection_item.result_text = str(detection_item.attr) + " " + str(detection_item.confidence*100) + "%"
        detection_result_list.append(detection_item)
    return detection_result_list

def GenerateTopNClassifyResult(resultList, n):
    '''
    processes classification result, returns top n categories

    Args:
        resultList: list, classification result
        n: integer, the quantity of categories with top confidece level user wants to obtain
    
    Returns:
        topNArray: numpy array, top n confidence
        confidenceIndex: numpy array, the corresponding index of top n confidence
    '''
    resultArray = resultList[0]
    confidenceList = resultArray[0, 0, 0, :]
    confidenceArray = np.array(confidenceList)
    confidenceIndex = np.argsort(-confidenceArray)
    topNArray = np.take(confidenceArray, confidenceIndex[0:n])
    return topNArray, confidenceIndex[0:n]

def FasterRCNNPostProcess(resultList, confidence_threshold):
    '''
    processes Faster RCNN inference result, returns a list of box coordinates

    Args:
        resultList: list, inference result
        confidence_threshold: float number, confidence threshold
    
    Returns:
        result_bbox: list, box coordinates
    '''
    tensor_num = resultList[0].reshape(-1)
    tensor_bbox = resultList.reshape(64, 304, 8)
    result_bbox = []
    for num in tensor_num:
        for bbox_idx in range(num):
            class_idx = attr * 2
            lt_x = tensor_bbox[class_idx][bbox_idx][0]
            lt_y = tensor_bbox[class_idx][bbox_idx][1]
            rb_x = tensor_bbox[class_idx][bbox_idx][2]
            rb_y = tensor_bbox[class_idx][bbox_idx][3]
            score = tensor_bbox[class_idx][bbox_idx][4]
            if score >= confidence_threshold:
                result_bbox.append([lt_x, lt_y, rb_x, rb_y, attr, score])
    return result_bbox