#Code Detect ที่ fix ขนาดจอที่ 640*640
import cv2
from ultralytics import YOLO

# Load Model ที่ผ่านการTrain(Yolov11) เรียบร้อย 
model = YOLO('yolo11x.pt') 

# รันการตรวจจับวัตถุบนวิดีโอจาก URL
results = model('https://www.youtube.com/watch?v=-kYZRNBE0vY', stream=True) #vid_stride=2, save=True, project='runs/detect/', name='exp')  
# stream=True เพื่อส่งข้อมูลเฟรมทีละเฟรม
# **หากไม่ต้องการให้Save ไฟล์ที่ detect** => ให้ลบtext หลัง save=True, ... ออกให้หมด

# กำหนดขนาดหน้าจอที่ต้องการแสดงผล
desired_size = (1280, 720)

# วนลูปเพื่อประมวลผลทีละเฟรม
for result in results:
    frame = result.orig_img  # ดึงภาพต้นฉบับจาก YOLO
    boxes = result.boxes  # Bounding Boxes
    classes = result.names  # ชื่อ Class

    # วาด Bounding Boxes ลงบนเฟรม
    annotated_frame = result.plot()

    # ปรับขนาดเฟรมให้อยู่ที่ 640x640
    resized_frame = cv2.resize(annotated_frame, desired_size)

    # แสดงผลด้วย OpenCV
    cv2.imshow('Detection', resized_frame)
    
    # กด 'q' เพื่อหยุดการแสดงผล
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดหน้าต่าง OpenCV เมื่อเสร็จสิ้น
cv2.destroyAllWindows()
