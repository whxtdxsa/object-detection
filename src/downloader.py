import json
import os
import subprocess

def prepare_dataset(dtype='val'):
    download_and_unzip_dataset(dtype=dtype)
    if not os.path.exists(f'./dataset/images/{dtype}'):
        mv(f'./dataset/images/{dtype}2017', f'./dataset/images/{dtype}')
        mv(f'./dataset/labels/annotations/instances_{dtype}2017.json', f'./dataset/labels/coco_{dtype}.json')
    if not os.path.exists(f'dataset/labels/{dtype}'):
        extract_person_data(f'./dataset/labels/coco_{dtype}.json', f'./dataset/labels/{dtype}.json')

def download_and_unzip_dataset(dataset_dir='./dataset', dtype='val'):
    if dtype != 'train' and dtype != 'val':
        print(f'Skipping downloaing {dtype} data: Incorrect dtype')
        return

    images_url = f'http://images.cocodataset.org/zips/{dtype}2017.zip'
    labels_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    zimages_dst = os.path.join(dataset_dir, f'images/{dtype}.zip') 
    zlabels_dst = os.path.join(dataset_dir, 'labels/annotations.zip')

    images_dst = os.path.dirname(zimages_dst) 
    labels_dst = os.path.dirname(zlabels_dst) 

    if os.path.exists(zimages_dst):
        print(f'Skipping Downloaing {dtype} Data: Already Exist')
    else:
        # Donwload Data
        os.makedirs(os.path.dirname(zimages_dst), exist_ok=True)
        os.makedirs(os.path.dirname(zlabels_dst), exist_ok=True)
        print(f'Donwload {dtype} Data')
        download_dataset(images_url, labels_url, zimages_dst, zlabels_dst) 
    
        # Unzip Data
        print(f'Unzip {dtype} Data')
        unzip_dataset(zimages_dst, zlabels_dst, images_dst, labels_dst) 

def download_dataset(images_url, labels_url, zimages_dst, zlabels_dst):
    subprocess.run(
        ['aria2c', '-x16', '-s16', '-d', os.path.dirname(zimages_dst), '-o', os.path.basename(zimages_dst), images_url],
        check=True,
        capture_output=True,
        text=True
    )
    subprocess.run(
        ['aria2c', '-x16', '-s16', '-d', os.path.dirname(zlabels_dst), '-o', os.path.basename(zlabels_dst), labels_url],
        check=True,
    )

def unzip_dataset(zimages_file, zlabels_file, images_dst, labels_dst):
    subprocess.run(
        ['7z', 'x', zimages_file, '-o'+images_dst],
        check=True,
    )
    subprocess.run(
        ['7z', 'x', zlabels_file, '-o'+labels_dst],
        check=True,
    )

def mv(dir, dst):
    subprocess.run(
        ['mv', dir, dst],
        check=True,
    )

def extract_person_data(original_file, new_file):
    with open(original_file, 'r') as f:
        coco = json.load(f)

    person_id = next(cat['id'] for cat in coco['categories'] if cat['name'] == 'person')

    filtered_anns = [ann for ann in coco['annotations'] if ann['category_id'] == person_id]
    valid_image_ids = set(ann['image_id'] for ann in filtered_anns)
    filtered_images = [img for img in coco['images'] if img['id'] in valid_image_ids]
    filtered_coco = {
        'images': filtered_images,
        'annotations': filtered_anns,
        'categories': [cat for cat in coco['categories'] if cat['id'] == person_id]
    }

    with open(new_file, 'w') as f:
        json.dump(filtered_coco, f)

if __name__ == '__main__':
    prepare_dataset('val') 


