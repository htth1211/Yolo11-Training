import os
import psycopg2
import torch
from datetime import datetime
from PIL import Image
import io
import cv2
import pathlib #แก้ PosixPath

# ฟังก์ชันสำหรับการบันทึกข้อมูลลง PostgreSQL
def save_to_postgresql(db_config, car_number, image_path, location):
    try:
        # เชื่อมต่อฐานข้อมูล
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # อ่านภาพและแปลงเป็นไบต์
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        # บันทึกข้อมูลลงในตาราง
        query = """
        INSERT INTO car_data (car_number, image_data, location)
        VALUES (%s, %s, %s);
        """
        cursor.execute(query, (car_number, image_data, location))
        conn.commit()

        # ปิดการเชื่อมต่อ
        cursor.close()
        conn.close()
        print(f"Data saved to PostgreSQL: {image_path}")
    except Exception as e:
        print(f"Error saving to PostgreSQL: {e}")


# ฟังก์ชันหลักสำหรับการตรวจจับและบันทึก
def detect_and_save(model_path, source_path, save_dir, db_config):
    # โหลดโมเดล YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # ตรวจสอบว่า directory สำหรับบันทึกผลลัพธ์มีอยู่หรือไม่
    os.makedirs(save_dir, exist_ok=True)

    # โหลดไฟล์รูปภาพ/วิดีโอ
    dataset = cv2.VideoCapture(source_path) if source_path.lower().endswith(('mp4', 'avi')) else [source_path]

    # ตรวจจับรูปภาพ/วิดีโอ
    for frame_idx, path_or_frame in enumerate(dataset):
        if isinstance(path_or_frame, str):  # กรณี source เป็นไฟล์รูปภาพ
            frame = cv2.imread(path_or_frame)
        else:  # กรณี source เป็นวิดีโอ
            ret, frame = path_or_frame.read()
            if not ret:
                break

        # รันโมเดล
        results = model(frame)

        # ดึงข้อมูลผลลัพธ์
        for box in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = box
            label = results.names[int(cls)]  # ดึงชื่อ class จาก YOLOv5
            car_number = label  # เช่น "car", "dog", หรือชื่อ class ที่คุณกำหนด


            # ครอบภาพที่ตรวจจับได้
            cropped_car = frame[int(y1):int(y2), int(x1):int(x2)]

            # สร้างชื่อไฟล์สำหรับบันทึกภาพ
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            image_path = os.path.join(save_dir, f"car_{timestamp}.jpg")
            cv2.imwrite(image_path, cropped_car)

            # บันทึกข้อมูลลงฐานข้อมูล
            location = "Unknown"  # คุณสามารถเพิ่มพิกัด GPS ได้
            save_to_postgresql(db_config, car_number, image_path, location)

        print(f"Detection complete for frame {frame_idx + 1}")

# ตั้งค่าการเชื่อมต่อฐานข้อมูล
db_config = {
    'dbname': '_____________',
    'user': '_____________',
    'password': '_____________',
    'host': '_____________',
    'port': '_____________'
}

# เรียกใช้งานฟังก์ชัน
if __name__ == "__main__":
    model_path = r"best.pt"  # Path ของโมเดลที่ฝึกแล้ว
    source_path = r"\path_input"  # Path ของไฟล์รูปภาพหรือวิดีโอ
    save_dir = r"\path_output"  # Directory สำหรับบันทึกภาพที่ตรวจจับได้
    
    temp = pathlib.PosixPath #แก้ PosixPath
    pathlib.PosixPath = pathlib.WindowsPath #แก้ PosixPath

    

    detect_and_save(model_path, source_path, save_dir, db_config)
    
