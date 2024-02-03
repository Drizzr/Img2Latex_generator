import os
import json
import tensorflow as tf
from model import Img2LaTex_model, Vocabulary, LatexProducer
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--beam", action='store_true',
                        default=False, help="use beam search instead of greedy decoding")
parser.add_argument("--beam_width", type=int, default=3)
parser.add_argument("--max_len", type=int, default=150)
    
args = parser.parse_args()

def load_from_checkpoint(args):
    """Load model from checkpoint"""
    print("loading model from checkpoint...")
    with open("checkpoint/chechpoint_epoch_26_0.0%_estimated_loss_0.27/params.json", "r") as f:
        params = json.load(f)

    vocab = Vocabulary("vocab.txt")

    vocab_size = vocab.n_tokens

    x = tf.random.uniform((1, 96, 480, 1))
    formula = tf.random.uniform((1, 1))

    model = Img2LaTex_model(embedding_dim=params["embedding_dim"], enc_out_dim=params["enc_out_dim"], vocab_size=vocab_size,
                            attention_head_size=params["attention_head_size"], encoder_units=params["encoder_units"],
                            decoder_units=params["decoder_units"],)
    
    
    model(x, formula)

    model.load_weights("checkpoint/chechpoint_epoch_26_0.0%_estimated_loss_0.27/weights.h5")
    print("model loaded successfully...")

    producer = LatexProducer(model, vocab, args.max_len)

    return producer


producer = load_from_checkpoint(args=args)


def process_img(img_path):
    img = img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1) / 255

    img_data = np.asarray(img, dtype=np.uint8) / 255  # convert to numpy array
    nnz_inds = np.where(img_data!=1) # returns tupel ([...], [...]) of indices where img_data is not 255

    if len(nnz_inds[0]) == 0:
        return img
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    img = img[y_min:y_max+1, x_min:x_max+1, :]
    
    width, height = img.shape[1], img.shape[0]

    if width / height < 480 / 96:
        new_width = 480 / 96 * height
        pad = (new_width - width) / 2
        img = tf.pad(img, [[0, 0], [math.ceil(pad), math.floor(pad)], [0, 0]], constant_values=1)
    
    elif width / height > 480 / 96:
        new_height = width * 96 / 480
        pad = (new_height - height) / 2
        img = tf.pad(img, [[math.ceil(pad), math.floor(pad)], [0, 0], [0, 0]], constant_values=1)

    img = tf.image.resize(img, (96, 480))

    img = tf.expand_dims(img, axis=0)
    
    #save image
    imgS = img[0, :, :, :]
    imgS = tf.image.convert_image_dtype(imgS, tf.uint8)
    imgS = tf.image.encode_png(imgS)
    tf.io.write_file("processed_img.png", imgS)

    return img

if __name__ == "__main__":
    while True:
        img_path = input("Enter image path: ")
        if img_path == "exit":
            break
        if not os.path.exists(img_path):
            print("Invalid path")
            continue
        try:
            img = process_img(img_path)
            if args.beam:
                latex = producer._beam_search(img, args.beam_width)
            else:
                latex = producer._greedy_decoding(img)
            print(latex)
        except KeyError as e:
            print(e)
            continue


