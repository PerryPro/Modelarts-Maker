import socket
import cv2
import numpy
import time
import hiai
from hiai.nn_tensor_lib import DataType
from atlasutil import camera, ai, presenteragent, dvpp_process
import datetime
def socket_client():
    # 建立sock连接
    # address定义要连接的服务器IP地址和端口号
    address = ('0.0.0.0', 23456)
    try:
        # 建立socket对象
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    camera_width = 1280
    camera_height = 720
	# 创建摄像头对象
    cap = camera.Camera(id=0, fps=20, width=camera_width, height=camera_height,
                        format=camera.CAMERA_IMAGE_FORMAT_YUV420_SP)
    if not cap.IsOpened():
        print("Open camera 0 failed")
        return

    dvpp_handle = dvpp_process.DvppProcess(camera_width, camera_height)


    # jpeg压缩参数
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    i = 5 #要发送的图像张数
    while i > 0:
        i = i-1
        # 停止1S 防止发送过快服务的处理不过来，如果服务端的处理很多，那么应该加大这个值
        time.sleep(1)
        sTime = datetime.datetime.now();
        yuv_img = cap.Read()#读取一帧图片
        orig_image = dvpp_handle.Yuv2Jpeg(yuv_img)
        yuv_img = yuv_img.reshape((1080, 1280))
        frame = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)


        # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
        # '.jpg'表示将图片按照jpg格式编码。
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        # 建立矩阵
        data = numpy.array(imgencode)
        # 将numpy矩阵转换成字符形式，以便在网络中传输
        stringData = data.tostring()

        # 先发送要发送的数据的长度
        # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        sock.send(str.encode(str(len(stringData)).ljust(16)));
        # 发送数据
        sock.send(stringData);
        # 读取服务器返回值
        receive = sock.recv(1024)
        eTime = datetime.datetime.now();
        print("time = ", (eTime - sTime).microseconds/1000," ms");
        if len(receive): #print(str(receive, encoding='utf-8'))
            print(receive.decode())
        # 读取下一帧图片
        # ret, frame = capture.read()
        if cv2.waitKey(10) == 27:
            break
    sock.close()


if __name__ == '__main__':
    socket_client()