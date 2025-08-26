#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py
LFW'yi indirir (sklearn), görüntüleri küçültür, gürültü ve/veya maske uygular,
ve diske klasör yapısıyla kaydeder:
out_dir/
  train/person_0/
  test/person_0/
...
"""

import os
import argparse
import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)

def add_noise_and_mask(img, noise_sigma=0.25, mask_prob=0.3):
    """img float32 [0,1], HxW"""
    noisy = img + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    if rng.random() < mask_prob:
        H, W = noisy.shape
        h = rng.integers(H//6, H//3)
        w = rng.integers(W//6, W//2)
        y0 = rng.integers(0, H-h)
        x0 = rng.integers(0, W-w)
        noisy[y0:y0+h, x0:x0+w] = 0.0
    return noisy

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_split(X, y, out_root, split_name, img_size, noise, mask_prob):
    H = W = img_size
    for idx, (img_rgb, label) in enumerate(zip(X, y)):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)/255.0
        small = add_noise_and_mask(small, noise_sigma=noise, mask_prob=mask_prob)
        # convert to 0..255 uint8
        small_u8 = (small*255.0).clip(0,255).astype(np.uint8)
        person_dir = os.path.join(out_root, split_name, f"person_{label:02d}")
        ensure_dir(person_dir)
        out_path = os.path.join(person_dir, f"img_{idx:05d}.png")
        cv2.imwrite(out_path, small_u8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out_data", help="çıktı klasörü kökü")
    ap.add_argument("--min-faces", type=int, default=50, help="LFW filtre: kişi başına minimum yüz")
    ap.add_argument("--img-size", type=int, default=16, help="çıktı çözünürlüğü")
    ap.add_argument("--noise", type=float, default=0.25, help="gaussian noise sigma")
    ap.add_argument("--mask-prob", type=float, default=0.30, help="maskelenmiş alan olasılığı")
    ap.add_argument("--test-size", type=float, default=0.2, help="test oranı")
    ap.add_argument("--people", type=str, nargs="+", default=None, help="spesifik isim listesi (LFW target_names)")
    args = ap.parse_args()

    print("[INFO] LFW verisi indiriliyor...")
    lfw = fetch_lfw_people(min_faces_per_person=args.min_faces, resize=1.0, color=True)
    X = lfw.images
    y = lfw.target
    names = lfw.target_names

    if args.people:
        keep = np.isin(names[y], args.people)
        X = X[keep]; y = y[keep]
        unique_names = sorted(list(set(names[y])))
        name_to_new = {nm:i for i, nm in enumerate(unique_names)}
        y = np.array([name_to_new[nm] for nm in names[y]], dtype=int)
        print(f"[INFO] Seçili kişiler: {unique_names}")
    else:
        # Otomatik en çok yüzü olan 3 kişi
        unique, counts = np.unique(y, return_counts=True)
        top3_idx = unique[np.argsort(-counts)][:3]
        keep = np.isin(y, top3_idx)
        X = X[keep]; y = y[keep]
        mapping = {old:i for i, old in enumerate(sorted(list(set(y))))}
        y = np.array([mapping[v] for v in y], dtype=int)
        print(f"[INFO] Otomatik seçilen 3 kişi")

    print(f"[INFO] Toplam görüntü: {len(X)} | Sınıf sayısı: {len(set(y))}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=SEED, stratify=y)

    out_root = args.out
    ensure_dir(out_root)
    save_split(Xtr, ytr, out_root, "train", args.img_size, args.noise, args.mask_prob)
    save_split(Xte, yte, out_root, "test", args.img_size, args.noise, args.mask_prob)
    print(f"[DONE] Kayıt tamamlandı. Klasör yapısı: {out_root}/train ve {out_root}/test")

if __name__ == "__main__":
    main()
