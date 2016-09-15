#!/usr/bin/env python3
import os
import sys
import argparse
import tensorflow as tf

from model import *
from utils.dataset import *
from utils.coco.pycocotools.coco import *

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default = 'train')
    parser.add_argument('--load', action = 'store_true', default = False)
    
    parser.add_argument('--train_image_dir', default = './train/images/')
    parser.add_argument('--train_caption_file', default = './train/captions_train2014.json')
    parser.add_argument('--train_annotation_file', default = './train/anns.csv')
    parser.add_argument('--train_info_file', default = './train/info.pickle')
    parser.add_argument('--train_conv5_feat_file', default = './train/conv5_feats.npz')
    parser.add_argument('--train_fc7_feat_file', default = './train/fc7_feats.npz')

    parser.add_argument('--val_image_dir', default = './val/images/')
    parser.add_argument('--val_caption_file', default = './val/captions_val2014.json')
    parser.add_argument('--val_info_file', default = './val/info.pickle')
    parser.add_argument('--val_conv5_feat_file', default = './val/conv5_feats.npz')
    parser.add_argument('--val_fc7_feat_file', default = './val/fc7_feats.npz')

    parser.add_argument('--test_image_dir', default = './test/images/')
    parser.add_argument('--test_info_file', default = './test/info.csv')
    parser.add_argument('--test_conv5_feat_file', default = './test/conv5_feats.npz')
    parser.add_argument('--test_fc7_feat_file', default = './test/fc7_feats.npz')
    parser.add_argument('--test_result_file', default = './test/results.csv')

    parser.add_argument('--word_table_file', default = './words/word_table.pickle')
    parser.add_argument('--glove_dir', default = './words/')
    parser.add_argument('--word2vec_scale', type = float, default = 0.5)
    parser.add_argument('--max_sent_len', type = int, default = 30)

    parser.add_argument('--save_dir', default = './models/')
    parser.add_argument('--val_period', type = int, default = 1)
    parser.add_argument('--save_period', type = int, default = 1)
    
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--num_epochs', type = int, default = 1000)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--weight_decay', type = float, default = 1e-4)

    parser.add_argument('--num_lstm', type = int, default = 1)
    parser.add_argument('--dim_hidden', type = int, default = 300)
    parser.add_argument('--dim_embed', type = int, default = 300)

    parser.add_argument('--init_embed_weight', action = 'store_true', default = False)
    parser.add_argument('--fix_embed_weight', action = 'store_true', default = False)
    parser.add_argument('--init_dec_weight', action = 'store_true', default = False)
    parser.add_argument('--fix_dec_weight', action = 'store_true', default = False)
    parser.add_argument('--init_dec_bias', action = 'store_true', default = False)
    parser.add_argument('--use_fc_feat', action = 'store_true', default = False)
    parser.add_argument('--batch_norm', action = 'store_true', default = False)  

    args = parser.parse_args()

    if args.mode == 'train':
        train_coco, train_data = prepare_train_data(args)
        val_coco, val_data = prepare_val_data(args)
    elif args.mode == 'val':
        val_coco, val_data = prepare_val_data(args)
    else:  
        test_data = prepare_test_data(args)

    with tf.Session() as sess:
        model = CaptionGenerator(args)
        sess.run(tf.initialize_all_variables())
        
        if args.mode == 'train':
            if args.load:
                model.load(sess)
            model.train_val(sess, train_coco, train_data, val_coco, val_data)
        elif args.mode == 'val':
            model.load(sess)
            model.val(sess, val_coco, val_data)
        else:
            model.load(sess)
            model.test(sess, test_data)

if __name__=="__main__":
     main(sys.argv)

