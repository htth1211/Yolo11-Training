#File Train moedel YOLOv11 Custom Dataset

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # ตรวจสอบว่า GPU พร้อมใช้งาน
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU instead.")

    # โหลดโมเดล
    model = YOLO("yolo11s.pt")  # โมเดล YOLO ที่เตรียมไว้

    # เริ่มการฝึก
    model.train(data="data.yaml",
            epochs=150,   # จำนวนรอบในการเทรน
            imgsz=640,    # ขนาดภาพที่ใช้เทรน
            batch=16,     # จำนวนตัวอย่างข้อมูลที่ใช้ประมวลผลในแต่ละรอบขณะฝึกฝนโมเดล เหมาะกับ GPU ที่มี VRAM จำกัด
            device=0,     # device=0 ใช้ GPU หมายเลข 0
            patience=0,    # ปิด Early stop
            project="run\detect",  #ตำแหน่งที่จะsave ตัวโมเดลที่เทรนเสร็จสมบูรณ์
            name="Train")          # ชื่อโฟลเดอร์ย่อยสำหรับโมเดลที่เทรนเสร็จสมบูรณ์




