import math
import os
import tensorflow as tf
import numpy as np

from base_model import *
from utils.nn import *

class CaptionGenerator(BaseModel):
    def build(self):
        print("Building the model ...")
        params = self.params
        
        num_lstm = params.num_lstm
        dim_embed = params.dim_embed
        dim_hidden = params.dim_hidden
        batch_size = params.batch_size         # n
        num_ctx = 196                          # k
        dim_ctx = 512                          # d
        dim_fc_feats = 4096
        bn = params.batch_norm      
        num_words = self.word_table.num_words
        max_sent_len = params.max_sent_len
        word2vec_scale = params.word2vec_scale

        contexts = tf.placeholder(tf.float32, [batch_size, num_ctx, dim_ctx])
        fc_feats = tf.placeholder(tf.float32, [batch_size, dim_fc_feats])
        sentences = tf.placeholder(tf.int32, [batch_size, max_sent_len])
        masks = tf.placeholder(tf.float32, [batch_size, max_sent_len])        
        is_train = tf.placeholder(tf.bool)

        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] for i in range(num_words)]) * word2vec_scale
        if params.init_embed_weight:
            if params.fix_embed_weight:
                emb_w = tf.cast(tf.convert_to_tensor(idx2vec), tf.float32)
            else:
                emb_w = weight('emb_w', [num_words, dim_embed], init_val=idx2vec)
        else:
            emb_w = weight('emb_w', [num_words, dim_embed], init='he')

        vec2idx = idx2vec.transpose()
        if params.init_dec_weight:
            if params.fix_dec_weight:
                dec_w = tf.cast(tf.convert_to_tensor(vec2idx), tf.float32)          
            else:
                dec_w = weight('dec_w', [dim_embed, num_words], init_val=vec2idx)
        else:
            dec_w = weight('dec_w', [dim_embed, num_words], init='he')
    
        if params.init_dec_bias: 
            dec_b = bias('dec_b', [num_words], init_val=self.word_table.word_freq)
        else:
            dec_b = bias('dec_b', [num_words], init_val=0.0)

        context_mean = tf.reduce_mean(contexts, 1)

        if params.use_fc_feat:
            init_feats = fc_feats
        else:
            init_feats = context_mean

        lstm = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)        

        if num_lstm == 1:
            memory = fully_connected(init_feats, dim_hidden, 'init_w1', 'init_b1', is_train, bn, 'tanh')
            output = fully_connected(init_feats, dim_hidden, 'init_w2', 'init_b2', is_train, bn, 'tanh')
            state = tf.nn.rnn_cell.LSTMStateTuple(memory, output)                   # state = (memory, output)
        else:
            memory1 = fully_connected(init_feats, dim_hidden, 'init_w11', 'init_b11', is_train, bn, 'tanh')
            output1 = fully_connected(init_feats, dim_hidden, 'init_w12', 'init_b12', is_train, bn, 'tanh')
            memory2 = fully_connected(init_feats, dim_hidden, 'init_w21', 'init_b21', is_train, bn, 'tanh')
            output = fully_connected(init_feats, dim_hidden, 'init_w22', 'init_b22', is_train, bn, 'tanh')
            state1 = tf.nn.rnn_cell.LSTMStateTuple(memory1, output1)                 # state = (memory, output)
            state2 = tf.nn.rnn_cell.LSTMStateTuple(memory2, output)                  # state = (memory, output)

        word_emb = tf.zeros([batch_size, dim_embed])

        loss0 = 0.0
        results = []
        probs = []

        for idx in range(max_sent_len):

            if idx > 0:
                tf.get_variable_scope().reuse_variables()                           
                word_emb = tf.cond(is_train, lambda: tf.nn.embedding_lookup(emb_w, sentences[:, idx-1]), lambda: word_emb)
            
            context_flat = tf.reshape(contexts, [-1, dim_ctx])                                                 #(n*k,d)
            context_encode1 = fully_connected(context_flat, dim_ctx, 'att_w11', 'att_b1', is_train, bn, None)  #(n*k,d)
            context_encode2 = fully_connected_no_bias(output, dim_ctx, 'att_w12', is_train, bn, None)          #(n,d)
            context_encode2 = tf.tile(tf.expand_dims(context_encode2, 1), [1, num_ctx, 1])                     #(n,k,d)
            context_encode2 = tf.reshape(context_encode2, [-1, dim_ctx])                                       #(n*k,d)          
            context_encode = nonlinear(context_encode1 + context_encode2, 'tanh')                              #(n*k,d)   

            alpha = fully_connected(context_encode, 1, 'att_w2', 'att_b2', is_train, bn, None)                 #(n*k,1)
            alpha = tf.reshape(alpha, [-1, num_ctx])                                                           #(n,k)
            alpha = tf.nn.softmax(alpha)                                                                       #(n,k)

            weighted_context = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), 1)
            
            if num_lstm == 1:
                with tf.variable_scope("lstm"):
                    output, state = lstm(tf.concat(1, [weighted_context, word_emb]), state)
            else:
                with tf.variable_scope("lstm1"):
                    output1, state1 = lstm(weighted_context, state1)

                with tf.variable_scope("lstm2"):
                    output, state2 = lstm(tf.concat(1, [word_emb, output1]), state2)
            
            expanded_output = tf.concat(1, [output, weighted_context, word_emb])
            logits1 = fully_connected(expanded_output, dim_embed, 'dec_w1', 'dec_b1', is_train, bn, 'relu')
            logits1 = dropout(logits1, 0.5, is_train)

            logits2 = tf.nn.xw_plus_b(logits1, dec_w, dec_b)
            if bn:
                logits2 = batch_norm(logits2, 'batch_norm_dec_w_dec_b', is_train)

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, sentences[:, idx])
            cross_entropy = cross_entropy * masks[:, idx]
            loss0 +=  tf.reduce_sum(cross_entropy)

            max_prob_word = tf.argmax(logits2, 1)
            results.append(max_prob_word)

            prob = tf.nn.softmax(logits2)
            max_prob = tf.reduce_max(prob, 1)
            probs.append(max_prob)

            word_emb = tf.cond(is_train, lambda: word_emb, lambda: tf.nn.embedding_lookup(emb_w, max_prob_word))

        results = tf.pack(results, axis=1)
        probs = tf.pack(probs, axis=1)

        loss0 = loss0 / tf.reduce_sum(masks)
        loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2'))
        loss = loss0 + loss1

        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        opt_op = optimizer.minimize(loss, global_step = self.global_step)

        self.contexts = contexts
        self.fc_feats = fc_feats
        self.sentences = sentences
        self.masks = masks
        self.is_train = is_train

        self.loss = loss
        self.loss0 = loss0
        self.loss1 = loss1
        self.opt_op = opt_op
        self.results = results
        self.probs = probs
        
        print("Model built.")        

    def get_feed_dict(self, batch, is_train):
        contexts, fc_feats, sents, masks, _ = batch
        return {self.contexts: contexts, self.fc_feats: fc_feats, self.sentences: sents, self.masks: masks, self.is_train: is_train}

