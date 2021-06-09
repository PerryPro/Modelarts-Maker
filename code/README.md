
# Retina模型推理：
用于RetinaNet目标检测算法的推理代码       
【由于Edge与Surface采用的目标检测算法相同，故调用模型进行推理的代码是可以通用的，只有模型文件和index文件不同】       
       
## Surface：用于水面平台图像的推理
- mark_detection.py：从test_data文件夹读取图片，将推理后的结果图片保存至output文件夹       
- index：规定目标检测项目类型和名称       
- client.py：运行于水面平台，从摄像头读取图像，传输给server，并获取回传的目标检测结果       
- server.py：运行于岸边基站，接收从client来的图像数据并保存到本地，随后调用模型进行目标检测推理，将推理结果以json格式回传       
- convert-6194-retina.om：离线化后的模型文件       
- 其他文件：调用模型推理的必要文件       
       
## Edge：用于岸边基站图像的推理
mark_detection.py与index作用与上述Yolo的相同       
- edge-retinamk1-asend.om：离线化后的模型文件       
- 其他文件：调用模型推理的必要文件       
       
# 执行流程：
在岸边基站：    
1.执行Surface文件夹里的server.py，启动服务端      
2.执行【摄像机0取图片】文件夹里的main.py，获取岸边基站摄像头拍摄的图片并保存到本地，随后执行Edge文件夹里的mark_detection.py进行推理。     
        
在水面平台：       
执行Surface文件夹里的client.py，传输图像到岸边基站，并获取从岸边基站回传的目标检测结果    


 
