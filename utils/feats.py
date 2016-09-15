import math
import caffe
import numpy as np
import cv2
import skimage
import skimage.io
from scipy.sparse import *
from tqdm import tqdm

vgg_deploy = '/home/aioria/caffemodels/vgg16.prototxt'
vgg_model = '/home/aioria/caffemodels/vgg16.caffemodel'
vgg_mean = '/home/aioria/caffemodels/ilsvrc_2012_mean.npy'

class FeatureExtractor(caffe.Net):
    def __init__(self, deploy=vgg_deploy, model=vgg_model, mean=vgg_mean, scale_dim=[224, 224], image_dim=[224, 224], isotropic=False):
        caffe.set_mode_gpu()
        caffe.Net.__init__(self, deploy, model, caffe.TEST)

        self.scale_dim = np.array(scale_dim)
        self.image_dim = np.array(image_dim)
        self.isotropic = isotropic

        self.transformer = caffe.io.Transformer({'data':self.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.load(mean).mean(1).mean(1))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2,1,0))
    
    def load_image(self, image_dir):
        image = skimage.img_as_float(skimage.io.imread(image_dir)).astype(np.float32)
        assert image.ndim == 2 or image.ndim == 3
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            image = np.tile(image, (1, 1, 3))
        elif image.shape[2] > 3:
            image = image[:, :, :3]
        return image

    def resize_image(self, image):
        a, b, _ = image.shape
        c, d = self.scale_dim
        if self.isotropic:
            if (c*1.0/a) <= (d*1.0/b):
                e = int(a*1.0*d/b)
                return cv2.resize(image, (e, d)) 
            else:          
                f = int(b*1.0*c/a)
                return cv2.resize(image, (c, f)) 
        else:
            return cv2.resize(image, (c, d))
    
    def crop_image(self, image):
        offset = ((image.shape[:2] - self.image_dim)/2.0).astype(np.int32)
        return image[offset[0]:offset[0]+self.image_dim[0], offset[1]:offset[1]+self.image_dim[1]]

    def load_and_preprocess_image(self, image_dir):
        return self.crop_image(self.resize_image(self.load_image(image_dir)))

    def get_features(self, image_list, batch_size=200, layers='conv5_3', layer_sizes=[512, 14, 14]):
        image_count = len(image_list)       
        num_batches = int(math.ceil(image_count * 1.0 / batch_size))

        assert len(layer_sizes)==1 or len(layer_sizes)==3

        for k in tqdm(list(range(num_batches)), desc='batch'):
            start = k * batch_size
            end = min(start + batch_size, image_count)
            current_batch = image_list[start:end]
            current_images = np.array([self.load_and_preprocess_image(x) for x in current_batch])
            caffe_in = np.zeros(np.array(current_images.shape)[[0, 3, 1, 2]], dtype=np.float32)

            for idx, in_ in enumerate(current_images):
                caffe_in[idx] = self.transformer.preprocess('data', in_)

            out = self.forward_all(blobs=[layers], **{'data':caffe_in})
            current_feats = out[layers]

            if len(layer_sizes) > 1:
               current_feats = current_feats.reshape(-1, layer_sizes[0], np.prod(layer_sizes[1:])).swapaxes(1,2)   

            sparse_cur_feats = to_sparse_csr(current_feats)

            if k == 0:
                feats = sparse_cur_feats
            else:
                feats = vstack((feats, sparse_cur_feats))

        return feats

def to_sparse_csr(feats):
    flat_feats = np.reshape(feats, (feats.shape[0], -1))
    return csr_matrix(flat_feats)

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

