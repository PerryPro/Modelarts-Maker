import hiai
from hiai.nn_tensor_lib import DataType
from atlasutil import camera, ai, presenteragent, dvpp_process
import cv2 as cv


def main():
    camera_width = 1280
    camera_height = 720
    cap = camera.Camera(id = 0, fps = 20, width = camera_width, height = camera_height , format = camera.CAMERA_IMAGE_FORMAT_YUV420_SP)
    if not cap.IsOpened():
        print("Open camera 0 failed")
        return

    dvpp_handle = dvpp_process.DvppProcess(camera_width, camera_height)
    i = 10
    while (i > 1):
        yuv_img = cap.Read()
        orig_image = dvpp_handle.Yuv2Jpeg(yuv_img)
        yuv_img = yuv_img.reshape((1080, 1280))
        img = cv.cvtColor(yuv_img, cv.COLOR_YUV2RGB_I420)
  #      img = cv.resize(img, (300, 300))
        fn = "test"+str(i)+".jpg"
        #fo = open(fn, "w")
        cv.imwrite(fn, img)
        i=i-1
        

  
if __name__ == "__main__":
    main()
