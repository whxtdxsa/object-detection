from collections import Counter, defaultdict
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def extract_annotations(annos):
    print('\nInfo of Annotations------------------')
    print('-------------------------------------')
    print(annos.keys())
    print(annos['images'][0])
    print(annos['annotations'][0])
    print(annos['categories'][0])

def plot_eda(annos, save_dir='./asset'):
    print('\nPlotting EDA Graphs------------------')
    print('-------------------------------------')
    os.makedirs(save_dir, exist_ok=True)
    plot_img_sizes(annos, os.path.join(save_dir, 'plot_img_sizes.png'))
    plot_bbox_areas(annos, os.path.join(save_dir, 'plot_bbox_areas.png'))
    plot_bbox_counts(annos, os.path.join(save_dir, 'plot_bbox_counts.png'))

def plot_img_sizes(annos, fdst):
    img_sizes = []
    for img in annos['images']:
        img_sizes.append((img['width'], img['height']))

    widths, heights = zip(*img_sizes)

    plt.figure(figsize=(10, 5))
    plt.scatter(widths, heights, alpha=0.3)

    plt.title('Image Size Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fdst)
    plt.clf()

def plot_bbox_areas(annos, fdst):
    bbox_areas = []
    for anno in annos['annotations']:
        bbox = anno['bbox']
        bbox_areas.append(bbox[2] * bbox[3])

    plt.figure(figsize=(10, 5))
    plt.hist(bbox_areas, bins=20)

    plt.title('Bounding Box Area Distribution')
    plt.xlabel('Area')
    plt.ylabel('Count')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fdst)
    plt.clf()

def plot_bbox_counts(annos, fdst):
    bbox_counts = Counter()
    for anno in annos['annotations']:
        bbox_counts[anno['image_id']] += 1

    plt.figure(figsize=(10, 5))
    plt.hist(bbox_counts.values(), bins=20)

    plt.title('Number of Objects per Image')
    plt.xlabel('Objects')
    plt.ylabel('Images')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fdst)
    plt.clf()

def draw_bbox(image_file, annos, fdst='./asset/bbox.png'):
    os.makedirs('./asset', exist_ok=True)
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    for anno in annos:  # [x1, y1, bw, bh]
        x1, y1, bw, bh = anno['bbox']
        x_max = x1 + bw
        y_max = y1 + bh
        draw.rectangle([x1, y1, x_max, y_max], outline='red', width=2)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fdst, bbox_inches='tight', pad_inches=0)
    plt.clf()

def get_image_id_to_image(annos):
    return {img['id']: img for img in annos['images']}

def get_image_id_to_annos(annos):
    image_id_to_annos = defaultdict(list)

    for anno in annos['annotations']:
        image_id_to_annos[anno['image_id']].append(anno)
    
    return image_id_to_annos
