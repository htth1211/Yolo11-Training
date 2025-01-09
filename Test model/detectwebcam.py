import psycopg2
from ultralytics import YOLO
import os

# ข้อมูลเชื่อมต่อ PostgreSQL
db_config = {
    'dbname': '___________',
    'user': '___________',
    'password': '___________',
    'host': '___________',
    'port': '___________'
}

# ฟังก์ชันสำหรับเชื่อมต่อฐานข้อมูล PostgreSQL
def connect_to_db(config):
    try:
        conn = psycopg2.connect(**config)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# ฟังก์ชันในการบันทึกผลการตรวจจับลงใน PostgreSQL
def save_detection_to_db(conn, image_path, class_name, confidence, xmin, ymin, xmax, ymax):
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO detections (image_path, class_name, confidence, xmin, ymin, xmax, ymax)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (image_path, class_name, confidence, xmin, ymin, xmax, ymax))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Error saving detection data: {e}")

# โหลดโมเดล YOLO
model = YOLO('best_abc.pt')  # เปลี่ยนเป็น path ไฟล์ .pt ที่เทรนมาแล้ว

# กำหนดที่อยู่ไฟล์ภาพหรือวิดีโอที่ต้องการตรวจจับ
source = '0'  # หรือใช้วิดีโอได้

# ทำการทำนายจากไฟล์ภาพ/วิดีโอ
results = model(source, show=True)

# เชื่อมต่อฐานข้อมูล PostgreSQL
conn = connect_to_db(db_config)

if conn:
    # เก็บข้อมูลการตรวจจับในฐานข้อมูล
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)  # คลาสของวัตถุ
            class_name = result.names[class_id]  # ชื่อคลาส
            confidence = box.conf  # ความมั่นใจ
            xmin, ymin, xmax, ymax = box.xyxy[0]  # ขอบเขตของ bounding box (xmin, ymin, xmax, ymax)

            # บันทึกผลการตรวจจับลงในฐานข้อมูล
            save_detection_to_db(conn, source, class_name, confidence, int(xmin), int(ymin), int(xmax), int(ymax))

    # ปิดการเชื่อมต่อฐานข้อมูล
    conn.close()
else:
    print("Failed to connect to the database.")
