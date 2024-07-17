import re
import time
from PIL import Image
import numpy as np
import pandas as pd
import torch
import clip
from transformers import CLIPTokenizer, CLIPModel
import torchtext

df = pd.read_json("./data/train.json")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_clip, preprocess = clip.load("ViT-B/32", device)

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

def get_clip_embedding_txt(text):
    text = clip.tokenize(text).to(device)
    # 埋め込みベクトルの取得
    with torch.no_grad():
        text_feature = model_clip.encode_text(text)

    return text_feature

def get_clip_embedding_img(image):
    image = preprocess(image).unsqueeze(0).to(device)
    # 埋め込みベクトルの取得
    with torch.no_grad():
        image_feature = model_clip.encode_image(image)

    return image_feature


# 埋め込みをバッチごとに保存する関数
def save_embeddings_txt(embeddings, batch_num):
    np.save(f'train_embeddings_txt_batch_{batch_num}.npy', embeddings)

def save_embeddings_img(embeddings, batch_num):
    np.save(f'train_embeddings_img_batch_{batch_num}.npy', embeddings)

# データフレームの各行(question)に対してCLIP embeddingを取得して置き換える
embeddings_list_txt = []
batch_size = 2500
batch_num = 0
start_time = time.time()

for i, text in enumerate(df['question']):
    text = process_text(text)
    embeddings = get_clip_embedding_txt(text)
    embeddings = embeddings.cpu()
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

# データフレームの各行(image)に対してCLIP embeddingを取得して置き換える
embeddings_list_img = []
batch_size = 2500
batch_num = 0

for i, image in enumerate(df['image']):
    image = Image.open(f"./data/train/{image}")
    embeddings = get_clip_embedding_img(image)
    embeddings = embeddings.cpu()
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

import glob
# .npy ファイルをすべて読み込む
npy_files = sorted(glob.glob('train_embeddings_txt_batch_*.npy'))

# 全埋め込みデータを格納するリスト
all_embeddings = []

# .npy ファイルを順番に読み込み、リストに追加
for file in npy_files:
    embeddings = np.load(file)
    for embedding in embeddings:
        all_embeddings.append(embedding)

# データフレームの 'embedding' 列に順に格納
df['clip_question'] = all_embeddings[:len(df)]
# 埋め込みを文字列に変換（カンマ区切り）
df['clip_question'] = df['clip_question'].apply(lambda x: ','.join(map(str, x.flatten())))

# =============================================================================================
# image
npy_files = sorted(glob.glob('train_embeddings_img_batch_*.npy'))

# 全埋め込みデータを格納するリスト
all_embeddings = []

# .npy ファイルを順番に読み込み、リストに追加
for file in npy_files:
    embeddings = np.load(file)
    for embedding in embeddings:
        all_embeddings.append(embedding)

# データフレームの 'embedding' 列に順に格納
df['clip_image'] = all_embeddings[:len(df)]
# 埋め込みを文字列に変換（カンマ区切り）
df['clip_image'] = df['clip_image'].apply(lambda x: ','.join(map(str, x.flatten())))

df.to_csv('train_emb.csv')
# validについては上記のtrain.jsonとtrainの部分を
# valid.jsonやvalidに変更して再度実行することでvalid_emb.csvが得られる
