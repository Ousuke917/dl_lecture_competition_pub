import re
import time
from PIL import Image
import numpy as np
import pandas as pd
import torch
import requests
from transformers import AutoProcessor, BlipModel
import torchtext

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def process_text(text):
    # lowercase
    text = text.lower()

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_blip_txt_embedding(text):
    text = process_text(text)
    inputs = processor(text=text, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return text_features

def get_blip_img_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs).to(device)
    return image_features

def save_embeddings_txt(embeddings, batch_num):
    np.save(f'/kaggle/working/train_embeddings_txt_batch_{batch_num}.npy', embeddings)

def save_embeddings_img(embeddings, batch_num):
    np.save(f'/kaggle/working/train_embeddings_img_batch_{batch_num}.npy', embeddings)

df = pd.read_csv('/kaggle/input/vqa-data2/train_emb.csv')

embeddings_list_txt = []
batch_size = 2500
batch_num = 0
start_time = time.time()

for i, text in enumerate(df['question']):
    embeddings = get_blip_txt_embedding(text)
    embeddings = embeddings.detach().cpu().numpy()
    embeddings_list_txt.append(embeddings)

    if (i + 1) % 500 == 0:
        print(f'finished:{i+1}/{len(df)},    処理時間:{time.time()-start_time}')

    # 5000個の埋め込みごとに保存
    if (i + 1) % batch_size == 0:
        save_embeddings_txt(embeddings_list_txt, batch_num)
        embeddings_list_txt = []  # リストをリセット
        batch_num += 1

# 最後のバッチが5000未満の場合でも保存
if embeddings_list_txt:
    save_embeddings_txt(embeddings_list_txt, batch_num)

embeddings_list_img = []
batch_size = 2500
batch_num = 0

for i, image in enumerate(df['image']):
    image = Image.open(f"/kaggle/input/vqa-data/train/train/{image}").to(device)
    embeddings = get_blip_img_embedding(image)
    embeddings = embeddings.detach().cpu().numpy()
    embeddings_list_img.append(embeddings)

    if (i + 1) % 500 == 0:
        print(f'finished:{i+1}/{len(df)},    処理時間:{time.time()-start_time}')

    # 5000個の埋め込みごとに保存
    if (i + 1) % batch_size == 0:
        save_embeddings_img(embeddings_list_img, batch_num)
        embeddings_list_img = []  # リストをリセット
        batch_num += 1
        
# 最後のバッチが5000未満の場合でも保存
if embeddings_list_img:
    save_embeddings_img(embeddings_list_img, batch_num)

##text
# .npy ファイルをすべて読み込む
npy_files = sorted(glob.glob('/kaggle/working/train_embeddings_txt_batch_*.npy'))

# 全埋め込みデータを格納するリスト
all_embeddings = []

# .npy ファイルを順番に読み込み、リストに追加
for file in npy_files:
    embeddings = np.load(file)
    for embedding in embeddings:
        all_embeddings.append(embedding)

# データフレームの 'embedding' 列に順に格納
df['blip_question'] = all_embeddings[:len(df)]
# 埋め込みを文字列に変換（カンマ区切り）
df['blip_question'] = df['blip_question'].apply(lambda x: ','.join(map(str, x.flatten())))

## image
npy_files = sorted(glob.glob('/kaggle/working/train_embeddings_img_batch_*.npy'))

# 全埋め込みデータを格納するリスト
all_embeddings = []

# .npy ファイルを順番に読み込み、リストに追加
for file in npy_files:
    embeddings = np.load(file)
    for embedding in embeddings:
        all_embeddings.append(embedding)

# データフレームの 'embedding' 列に順に格納
df['blip_image'] = all_embeddings[:len(df)]
# 埋め込みを文字列に変換（カンマ区切り）
df['blip_image'] = df['blip_image'].apply(lambda x: ','.join(map(str, x.flatten())))
df.head()
df.to_csv('/kaggle/working/train_emb.csv')
