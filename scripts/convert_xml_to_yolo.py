import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

CLASS_MAP = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
}

DATA_DIR = os.path.expanduser("~/Desktop/mca-yolo-reproduce/data/China_MotorBike/train")
OUTPUT_DIR = os.path.expanduser("~/Desktop/mca-yolo-reproduce/data/yolo_format")

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

def convert_xml_to_yolo(xml_path, output_txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))
    return len(lines)

def main():
    random.seed(42)
    annotations_dir = os.path.join(DATA_DIR, "annotations/xmls")
    images_dir = os.path.join(DATA_DIR, "images")
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    print(f"找到 {len(xml_files)} 个标注文件")

    class_counts = defaultdict(int)
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name in CLASS_MAP:
                class_counts[name] += 1
    print("\n类别分布：")
    for name, count in sorted(class_counts.items()):
        print(f"  {name}: {count}")

    random.shuffle(xml_files)
    total = len(xml_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    train_files = xml_files[:train_end]
    val_files = xml_files[train_end:val_end]
    test_files = xml_files[val_end:]
    print(f"\n训练 {len(train_files)}，验证 {len(val_files)}，测试 {len(test_files)}")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    for split_name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        img_out_dir = os.path.join(OUTPUT_DIR, split_name, "images")
        label_out_dir = os.path.join(OUTPUT_DIR, split_name, "labels")
        for xml_file in file_list:
            stem = xml_file.replace(".xml", "")
            src_img = os.path.join(images_dir, f"{stem}.jpg")
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(img_out_dir, f"{stem}.jpg"))
            xml_path = os.path.join(annotations_dir, xml_file)
            txt_path = os.path.join(label_out_dir, f"{stem}.txt")
            convert_xml_to_yolo(xml_path, txt_path)

    print(f"\n✅ 完成！YOLO 格式保存在：{OUTPUT_DIR}")
    for split in ["train", "val", "test"]:
        n = len(os.listdir(os.path.join(OUTPUT_DIR, split, "images")))
        print(f"  {split}: {n} 张")

if __name__ == "__main__":
    main()
