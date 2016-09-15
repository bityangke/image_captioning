import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.words import *
from utils.dataset import *
from utils.coco.pycocotools.coco import *
from utils.coco.pycocoevalcap.eval import *

class BaseModel(object):
    def __init__(self, params):
        self.params = params
        self.save_dir = params.save_dir
        self.batch_size = params.batch_size

        self.word_table = WordTable(params.dim_embed, params.max_sent_len, params.word_table_file)
        self.word_table.load()

        with tf.variable_scope('caption_generator'):
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
            self.build()
            self.saver = tf.train.Saver(max_to_keep = 100)

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def train_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train = True)
        return sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict = feed_dict)

    def val_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train = False)
        return sess.run([self.results, self.probs], feed_dict = feed_dict)

    def test_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train = False)
        return sess.run([self.results, self.probs], feed_dict = feed_dict)

    def train_val(self, sess, train_coco, train_data, val_coco, val_data):
        print("Training the model ...")
        params = self.params
        num_epochs = params.num_epochs

        for epoch_no in tqdm(list(range(num_epochs)), desc='epoch'):
            for idx in tqdm(list(range(train_data.num_batches)), desc='batch'):

                batch = train_data.next_batch()
                _, loss0, loss1, global_step = self.train_batch(sess, batch)
                print("Loss0=%f Loss1=%f" %(loss0, loss1))

            train_data.reset()

            if (epoch_no + 1) % params.save_period == 0:
                self.save(sess)

            if (epoch_no + 1) % params.val_period == 0:
                self.val(sess, val_coco, val_data)

        print("Training complete.")

    def val(self, sess, val_coco, val_data):
        print("Validating the model ...")
        results = []
        count = val_data.real_count

        for k in tqdm(list(range(val_data.num_batches)), desc='batch'):

            batch = val_data.next_batch()
            _, _, _, _, img_ids = batch
            res, _ = self.val_batch(sess, batch)
            real_size = min(self.batch_size, count-k*self.batch_size)
            for idx in range(real_size):
                sent = self.word_table.indices_to_sent(res[idx])
                results.append({'image_id': img_ids[idx], 'caption': sent})

        val_data.reset() 

        val_res_coco = val_coco.loadRes2(results)
        scorer = COCOEvalCap(val_coco, val_res_coco)
        scorer.evaluate()
        print("Validation complete.")

    def test(self, sess, test_data):
        print("Testing the model ...")
        test_info_file = self.params.test_info_file
        test_result_file = self.params.test_result_file
        caps = []
        count = test_data.real_count
        
        for k in tqdm(list(range(test_data.num_batches)), desc='batch'):
            batch = test_data.next_batch()
            res, _ = self.test_batch(sess, batch)
            real_size = min(self.batch_size, count-k*self.batch_size)
            for idx in range(real_size):
                sent = self.word_table.indices_to_sent(res[idx])
                caps.append(sent)

        test_info = pd.read_csv(test_info_file)
        test_images = test_info['image'].values

        results = pd.DataFrame({'image':test_images, 'caption':caps})
        results.to_csv(test_result_file)
        print("Testing complete.")

    def save(self, sess):
        print(("Saving model to %s" % self.save_dir))
        self.saver.save(sess, self.save_dir, self.global_step)


    def load(self, sess):
        print("Loading model ...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
