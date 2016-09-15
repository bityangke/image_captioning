import os
import math
import numpy as np
import pandas as pd
import pickle
import skimage
import skimage.io

from .feats import *
from .words import *
from .coco.pycocotools.coco import *

class DataSet():
    def __init__(self, img_to_feat_ids, conv5_feats, fc7_feats, caps, masks, img_ids, batch_size, shuffle=True):
        self.img_to_feat_ids = img_to_feat_ids
        self.conv5_feats = conv5_feats
        self.fc7_feats = fc7_feats
        self.caps = caps
        self.masks = masks
        self.img_ids = img_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_sent_len = len(caps[0])
        self.setup()

    def setup(self):
        self.real_count = len(self.img_ids)
        self.num_batches = int(math.ceil(self.real_count*1.0/self.batch_size))
        self.count = self.batch_size * self.num_batches
        self.fake_count = self.count - self.real_count

        if self.fake_count > 0:
            fake_caps = np.zeros((self.fake_count, self.max_sent_len)).astype(np.int32)
            self.caps = np.concatenate((self.caps, fake_caps), 0)
            fake_masks = np.zeros((self.fake_count, self.max_sent_len))
            self.masks = np.concatenate((self.masks, fake_masks), 0)
            fake_img_ids = -np.ones([self.fake_count]).astype(np.int32)
            self.img_ids = np.concatenate((self.img_ids, fake_img_ids), 0)

        self.current_index = 0
        self.indices = list(range(self.count))
        self.reset()

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        assert self.has_next_batch()
        from_, to_ = self.current_index, self.current_index + self.batch_size
        cur_idx = self.indices[from_:to_]

        conv5_feats = []
        fc7_feats = []
        caps = []
        masks = []
        img_ids = []
        for i in cur_idx:
            caps.append(self.caps[i])
            masks.append(self.masks[i])
            img_ids.append(self.img_ids[i])
            if self.img_ids[i]==-1:
                cur_conv5_feats = np.zeros((196, 512))
                cur_fc7_feats = np.zeros((4096))
            else:
                cur_conv5_feats = self.conv5_feats[self.img_to_feat_ids[self.img_ids[i]]]
                cur_conv5_feats = np.array(cur_conv5_feats.todense()).squeeze().reshape(196, 512)
                cur_fc7_feats = self.fc7_feats[self.img_to_feat_ids[self.img_ids[i]]] 
                cur_fc7_feats = np.array(cur_fc7_feats.todense()).squeeze()
            conv5_feats.append(cur_conv5_feats)
            fc7_feats.append(cur_fc7_feats)
   
        self.current_index += self.batch_size
        return np.array(conv5_feats), np.array(fc7_feats), np.array(caps), np.array(masks), np.array(img_ids)

    def has_next_batch(self):
        return self.current_index + self.batch_size <= self.count


class DataSet2():
    def __init__(self, img_ids, conv5_feats, fc7_feats, batch_size, max_sent_len):
        self.img_ids = img_ids
        self.conv5_feats = conv5_feats
        self.fc7_feats = fc7_feats
        self.batch_size = batch_size
        self.max_sent_len = max_sent_len
        self.setup()

    def setup(self):
        self.real_count = len(self.img_ids)
        self.num_batches = int(math.ceil(self.real_count*1.0/self.batch_size))
        self.count = self.batch_size * self.num_batches
        self.fake_count = self.count - self.real_count

        self.caps = np.zeros((self.count, self.max_sent_len)).astype(np.int32)       # fake captions 
        self.masks = np.ones((self.count, self.max_sent_len))                        # fake masks         

        if self.fake_count > 0:
            fake_img_ids = -np.ones([self.fake_count]).astype(np.int32)
            self.img_ids = np.concatenate((self.img_ids, fake_img_ids), 0)          

        self.current_index = 0
        self.indices = list(range(self.count))
        self.reset()

    def reset(self):
        self.current_index = 0

    def next_batch(self):
        assert self.has_next_batch()
        from_, to_ = self.current_index, self.current_index + self.batch_size
        cur_idx = self.indices[from_:to_]

        conv5_feats = []
        fc7_feats = []
        caps = []
        masks = []
        img_ids = []
        for i in cur_idx:
            caps.append(self.caps[i])
            masks.append(self.masks[i])
            img_ids.append(self.img_ids[i])
            if self.img_ids[i]==-1:
                cur_conv5_feats = np.zeros((196, 512))
                cur_fc7_feats = np.zeros((4096))
            else:
                cur_conv5_feats = self.conv5_feats[i]
                cur_conv5_feats = np.array(cur_conv5_feats.todense()).squeeze().reshape(196, 512)
                cur_fc7_feats = self.fc7_feats[i] 
                cur_fc7_feats = np.array(cur_fc7_feats.todense()).squeeze()
            conv5_feats.append(cur_conv5_feats)
            fc7_feats.append(cur_fc7_feats)
   
        self.current_index += self.batch_size
        return np.array(conv5_feats), np.array(fc7_feats), np.array(caps), np.array(masks), np.array(img_ids)

    def has_next_batch(self):
        return self.current_index + self.batch_size <= self.count


def prepare_train_data(args):
    image_dir, caption_file, annotation_file = args.train_image_dir, args.train_caption_file, args.train_annotation_file
    info_file, conv5_feat_file, fc7_feat_file = args.train_info_file, args.train_conv5_feat_file, args.train_fc7_feat_file

    word_table_file, glove_dir = args.word_table_file, args.glove_dir
    dim_embed, batch_size, max_sent_len = args.dim_embed, args.batch_size, args.max_sent_len

    coco = COCO(caption_file)
    coco.filter_by_cap_len(max_sent_len)
    
    if not os.path.exists(annotation_file):
        annotations = process_captions(coco, annotation_file)
    else:
        annotations = pd.read_csv(annotation_file)

    image_ids = annotations['image_id'].values
    captions = annotations['caption'].values
    print("Number of training captions = %d" %(len(captions)))

    print("Building the word table ...")
    word_table = WordTable(dim_embed, max_sent_len, word_table_file)
    if not os.path.exists(word_table_file):
        word_table.load_glove(glove_dir)
        word_table.build(captions)
        word_table.save()
    else:
        word_table.load()
    print("Word table built. Number of words = %d." %(word_table.num_words))

    caps, masks = symbolize_captions(captions, word_table)

    img_to_feat_ids, _, conv5_feats, fc7_feats = extract_feats(coco, image_dir, info_file, conv5_feat_file, fc7_feat_file)

    print("Building the training dataset ...")
    dataset = DataSet(img_to_feat_ids, conv5_feats, fc7_feats, caps, masks, image_ids, batch_size)
    print("Dataset built.")
    return coco, dataset


def prepare_val_data(args):
    image_dir, caption_file = args.val_image_dir, args.val_caption_file
    info_file, conv5_feat_file, fc7_feat_file = args.val_info_file, args.val_conv5_feat_file, args.val_fc7_feat_file

    word_table_file, glove_dir = args.word_table_file, args.glove_dir
    dim_embed, batch_size, max_sent_len = args.dim_embed, args.batch_size, args.max_sent_len

    coco = COCO(caption_file)
    coco.filter_by_cap_len(max_sent_len)
   
    _, feat_to_img_ids, conv5_feats, fc7_feats = extract_feats(coco, image_dir, info_file, conv5_feat_file, fc7_feat_file)

    print("Building the validation dataset ...")
    dataset = DataSet2(feat_to_img_ids, conv5_feats, fc7_feats, batch_size, max_sent_len)
    print("Dataset built.")
    return coco, dataset


def prepare_test_data(args):
    image_dir, info_file = args.test_image_dir, args.test_info_file
    conv5_feat_file, fc7_feat_file = args.test_conv5_feat_file, args.test_fc7_feat_file

    word_table_file, glove_dir = args.word_table_file, args.glove_dir
    dim_embed, batch_size, max_sent_len = args.dim_embed, args.batch_size, args.max_sent_len

    #check_files(image_dir)
    files = os.listdir(image_dir)
    images = [os.path.join(image_dir, f) for f in files if f.lower().endswith('.jpg')]
    img_ids = np.array(list(range(len(images))))
    info = pd.DataFrame({'image': images, 'image_id': img_ids})
    info.to_csv(info_file)

    conv5_feats, fc7_feats = extract_feats2(images, conv5_feat_file, fc7_feat_file)

    print("Building the testing dataset ...")    
    dataset = DataSet2(img_ids, conv5_feats, fc7_feats, batch_size, max_sent_len)
    print("Dataset built.")
    return dataset

def check_files(image_dir):
    print("Checking image files in %s" %(image_dir))
    files = os.listdir(image_dir)
    images = [os.path.join(image_dir, f) for f in files if f.lower().endswith('.jpg')]
    good_imgs = []
    for img in images:
        try:
           x = skimage.img_as_float(skimage.io.imread(img)).astype(np.float32)
           good_imgs.append(img)
        except:
           print("Image %s is corrupted and will be removed." %(img))
           os.remove(img)
    good_files = [img.split(os.sep)[-1] for img in good_imgs]
    return good_files

def process_captions(coco, annotation_file):
    captions = [coco.anns[k]['caption'] for k in coco.anns]
    image_ids = [coco.anns[k]['image_id'] for k in coco.anns]
    annotations = pd.DataFrame({'image_id': image_ids, 'caption': captions})
    annotations.to_csv(annotation_file)
    return annotations


def symbolize_captions(captions, word_table):
    indices = []
    masks = []
    for cap in captions:
        idx, mask = word_table.symbolize_sent(cap)
        indices.append(idx)
        masks.append(mask)
    return np.array(indices), np.array(masks)


def extract_feats(coco, image_dir, info_file, conv5_feat_file, fc7_feat_file):
    image_ids = [coco.imgs[k]['id'] for k in coco.imgs]
    images = [os.path.join(image_dir, coco.imgs[k]['file_name']) for k in coco.imgs]
    print("%d images found in %s" %(len(images), image_dir))

    img_to_feat_id = {}
    feat_to_img_id = []
    for i, img_id in enumerate(image_ids):
        img_to_feat_id[img_id] = i
        feat_to_img_id.append(img_id)

    pickle.dump([img_to_feat_id, feat_to_img_id], open(info_file, 'wb'))

    if not os.path.exists(conv5_feat_file):
        print("Extracting Conv5_3 features from these images ...")
        vgg_net = FeatureExtractor()
        conv5_feats = vgg_net.get_features(images, layers = 'conv5_3', layer_sizes = [512, 14, 14])
        save_sparse_csr(conv5_feat_file, conv5_feats)
        print("Conv5_3 features extracted and saved.")
    else:
        print("Conv5_3 features already exist in %s" %(conv5_feat_file))
        print("Loading Conv5_3 features ...")
        conv5_feats = load_sparse_csr(conv5_feat_file)
        print("Conv5_3 features loaded.")

    if not os.path.exists(fc7_feat_file):
        vgg_net = FeatureExtractor()
        print("Extracting FC7 features from these images ...")
        fc7_feats = vgg_net.get_features(images, layers = 'fc7', layer_sizes = [4096])
        save_sparse_csr(fc7_feat_file, fc7_feats)
        print("FC7 features extracted and saved.")
    else:
        print("FC7 features already exist in %s" %(fc7_feat_file))
        print("Loading FC7 features ...")
        fc7_feats = load_sparse_csr(fc7_feat_file)
        print("FC7 features loaded.")

    return img_to_feat_id, feat_to_img_id, conv5_feats, fc7_feats


def extract_feats2(images, conv5_feat_file, fc7_feat_file):

    if not os.path.exists(conv5_feat_file):
        print("Extracting Conv5_3 features from the images ...")
        vgg_net = FeatureExtractor()
        conv5_feats = vgg_net.get_features(images, layers = 'conv5_3', layer_sizes = [512, 14, 14])
        save_sparse_csr(conv5_feat_file, conv5_feats)
        print("Conv5_3 features extracted and saved.")
    else:
        print("Conv5_3 features already exist in %s" %(conv5_feat_file))
        print("Loading Conv5_3 features ...")
        conv5_feats = load_sparse_csr(conv5_feat_file)
        print("Conv5_3 features loaded.")

    if not os.path.exists(fc7_feat_file):
        print("Extracting FC7 features from the images ...")
        vgg_net = FeatureExtractor()
        fc7_feats = vgg_net.get_features(images, layers = 'fc7', layer_sizes = [4096])
        save_sparse_csr(fc7_feat_file, fc7_feats)
        print("FC7 features extracted and saved.")
    else:
        print("FC7 features already exist in %s" %(fc7_feat_file))
        print("Loading FC7 features ...")
        fc7_feats = load_sparse_csr(fc7_feat_file)
        print("FC7 features loaded.")

    return conv5_feats, fc7_feats

