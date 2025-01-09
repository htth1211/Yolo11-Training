#ใช้สำหรับ detect แบบแสดงหน้าจอ(Bounding Box)

from ultralytics import YOLO
import cv2  # สำหรับการจัดการวิดีโอ (กรณีใช้งานเสริม)

# โหลดโมเดล YOLO
model = YOLO("best.pt")

#Path ของ source => (= 0, 1, ... คือ webcam ตัวที่ 1, 2, ...) ถ้าเป็น img,mp4,link ให้ใส่ path ของไฟล์นั้น

model.predict(source="0", 
            vid_stride=1,  # ใช้สำหรับ VDO,Webcam กำหนดจำนวนเฟรมที่ข้ามไปในแต่ละรอบ เช่น vid_stride=1 จะไม่ข้ามเฟรม
            show=True,     # จะทำให้แสดงผลลัพธ์บนจอ
            save=True, 
            project='runs/detect/', 
            name='exp')

#(save=True, project='_', name='_') คือ path ที่จะsave ไฟล์Detect *จะอยู่ที่เดียวกันกับไฟล์นี้*
# เช่นอันนี้คือ path => runs\detect\exp จะsave ไว้ในโฟลเดอร์=>exp


