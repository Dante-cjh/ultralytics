import json

def simple_fix_category_ids(json_file_path, output_file_path=None):

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("开始修复category_id...")

    # 修复category
    for category in data.get('categories', []):
        if 'id' in category:
            category['id'] -= 1

    # 修复annotations中的所有category_id
    fixed_count = 0
    for annotation in data.get('annotations', []):
        if 'category_id' in annotation:
            annotation['category_id'] -= 1
            fixed_count += 1
    
    print(f"修复；了 {fixed_count} 个标注的category_id")

    # 保存文件
    if output_file_path is None:
        output_json_path = json_file_path.replace('.json', '_fixed.json')
    else:
        output_json_path = output_file_path
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"修复后的文件保存为: {output_json_path}")

    # 验证
    print("\n验证修复结果：")
    category_ids = set()
    for annotation in data.get('annotations', []):
        if 'category_id' in annotation:
            category_ids.add(annotation['category_id'])

    print(f"现在的category_id范围为: {min(category_ids) if category_ids else 'None'} - {max(category_ids) if category_ids else 'None'}")

    return data


if __name__ == "__main__":
    input_json_path = "/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/test/annotation_coco.json"
    output_file_path = "/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/test/annotation_coco_fixed.json"

    simple_fix_category_ids(input_json_path, output_file_path)

