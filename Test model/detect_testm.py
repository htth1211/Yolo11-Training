from ultralytics import YOLO

model = YOLO('best.pt') #เปลี่ยนเป็น path ไฟล์ .pt ที่เทรนมาแล้ว

#results = model(__,show= True), __คือ sorce เช่น 0 คือกล้องตัวแรก
#หากให้ source เป็นimg,vdo,link ให้ใส่ '' แล้วใส่ path ของไฟล์ที่เราต้องการ
results = model('https://www.youtube.com/watch?v=pcxT69UBq9I',show= False,  save=True, project='runs/detect/', name='exp') 
#project=__, name=__ คือชื่อ folder ที่ต้องการให้จัดเก็บ (จะจัดเก็บอยู่ที่เดียวกันไฟล์นี้)
#ตัวอย่าง ข้างบนคือ path => 'C:\Users\WINDOWS\Documents\SmartShift\yolov11\runs\detect\exp'

for result in results:
    boxes = result.boxes
    classes = result.names
