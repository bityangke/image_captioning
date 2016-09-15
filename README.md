This is an image caption generator loosely based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015). It first uses the VGG16 CNN network (implemented in caffe) to extract the conv5_3 feature of the input image, then uses a LSTM RNN network (implemented in tensorflow) to decode this feature into a natural language sentence. A soft attention mechanism is included to improve the quality of the generated caption. I have also added the option to initialize the word embedding with GloVe (nlp.stanford.edu/projects/glove/), to initialize the LSTM with the fc7 feature of VGG16 network (instead of the mean conv5_3 feature as described in the paper), and to use a stack of two LSTMs instead of only one to produce the words, although these modifications do not appear to yield a significant improvement over the original model.
