#Packages and functions
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from skimage.segmentation import find_boundaries
import numpy as np
from numba import jit
from scipy import ndimage
from tqdm import tqdm, tqdm_notebook
import numpy as np
from numba import jit,njit
from scipy import ndimage
from tqdm import tqdm, tqdm_notebook
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import transform,metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Random rotation of an image by a multiple of 90 degrees

def random_90rotation( img ):
    return transform.rotate(img, 90*np.random.randint( 0, 5 ), preserve_range=True)

# Runtime data augmentation

def get_train_val_generators(X_train, Y_train, X_val,Y_val,
                             batch_size=32, seed=42, rotation_range=0,
                             horizontal_flip=True, vertical_flip=True,
                             width_shift_range=0.0,
                             height_shift_range=0.0,
                             shear_range=0.0,
                             brightness_range=None,
                             rescale=None,
                             preprocessing_function=None,
                             show_examples=False):
    X_test, Y_test = X_val,Y_val

    # Image data generator distortion options
    data_gen_args = dict( rotation_range = rotation_range,
                          width_shift_range=width_shift_range,
                          height_shift_range=height_shift_range,
                          shear_range=shear_range,
                          brightness_range=brightness_range,
                          preprocessing_function=preprocessing_function,
                          horizontal_flip=horizontal_flip,
                          vertical_flip=vertical_flip,
                          rescale = rescale,
                          fill_mode='reflect')


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)


    # Validation data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator(rescale=rescale)
    Y_datagen_val = ImageDataGenerator(rescale=rescale)
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=False, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=False, seed=seed)

    if show_examples:
        plt.figure(figsize=(10,10))
        # generate samples and plot
        for i in range(3):
            # define subplot
            plt.subplot(321 + 2*i)
            # generate batch of images
            batch = X_train_augmented.next()
            # convert to unsigned integers for viewing
            image = batch[0]
            # plot raw pixel data
            plt.imshow(image[:,:,0], vmin=0, vmax=1, cmap='gray')
            plt.subplot(321 + 2*i+1)
            # generate batch of images
            batch = Y_train_augmented.next()
            # convert to unsigned integers for viewing
            image = batch[0]
            # plot raw pixel data
            plt.imshow(image[:,:,0], vmin=0, vmax=1, cmap='gray')
        # show the figure
        plt.show()
        X_train_augmented.reset()
        Y_train_augmented.reset()

    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    test_generator = zip(X_test_augmented, Y_test_augmented)
    print("data augmentation: Done!")
    return train_generator, test_generator



def convert_to_oneHot(data, eps=1e-8):
    """
    Converts labelled images (`data`) to one-hot encoding.
    Parameters
    ----------
    data : array(int)
        Array of lablelled images.
    Returns
    -------
    data_oneHot : array(int)
        Array of one-hot encoded images.
    """
    data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(add_boundary_label(data[i].astype(np.int32)))
        if ( np.abs(np.max(data[i])) <= eps ):
            data_oneHot[i][...,0] *= 0

    return data_oneHot


def add_boundary_label(lbl, dtype=np.uint16):
    """
    Find boundary labels for a labelled image.
    Parameters
    ----------
    lbl : array(int)
         lbl is an integer label image (not binarized).
    Returns
    -------
    res : array(int)
        res is an integer label image with boundary encoded as 2.
    """

    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(dtype)
    res[b] = 2
    return res


def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """ n_classes will be determined by max lbl value if its value is None """
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot


def normalize(img, mean, std):
    """
    Mean-Std Normalization.
    Parameters
    ----------
    img : array(float)
        Array of source images.
    mean : float
        mean intensity of images.
    std: float
        standard deviation of intensity of images.
    Returns
    -------
    (img - mean)/std: array(float)
       Normalized images
    """
    return (img - mean) / std


def denormalize(img, mean, std):
    """
    Mean-Std De-Normalization.
    Parameters
    ----------
    img : array(float)
        Array of source images.
    mean : float
        mean intensity of images.
    std: float
        standard deviation of intensity of images.
    Returns
    -------
    img * std + mean: array(float)
        De-normalized images
    """
    return (img * std) + mean


def zero_out_train_data(X_train, Y_train, fraction):
    """
    Fractionates training data according to the specified `fraction`.
    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    fraction: float (between 0 and 100)
        fraction of training images.
    Returns
    -------
    X_train : array(float)
        Fractionated array of source images.
    Y_train : float
        Fractionated array of label images.
    """
    train_frac = int(np.round((fraction / 100) * X_train.shape[0]))
    Y_train[train_frac:] *= 0

    return X_train, Y_train


def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max() + 1, lab2.max() + 1), dtype=np.int)
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1
    return psg


def intersection_over_union(psg):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    rsum = np.sum(psg, 0, keepdims=True)
    csum = np.sum(psg, 1, keepdims=True)
    return psg / (rsum + csum - psg)


def matching_iou(psg, fraction=0.5):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    iou = intersection_over_union(psg)
    matching = iou > fraction
    matching[:, 0] = False
    matching[0, :] = False
    return matching

def measure_precision(iou=0.5, partial_dataset=False):
    def precision(lab_gt, lab, iou=iou, partial_dataset=partial_dataset):
        """
        precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching
        :Authors:
            Coleman Broaddus
        """
        psg = pixel_sharing_bipartite(lab_gt, lab)
        matching = matching_iou(psg, fraction=iou)
        assert matching.sum(0).max() < 2
        assert matching.sum(1).max() < 2
        n_gt = len(set(np.unique(lab_gt)) - {0})
        n_hyp = len(set(np.unique(lab)) - {0})
        n_matched = matching.sum()
        if partial_dataset:
            return n_matched, (n_gt + n_hyp - n_matched)
        else:
            return n_matched / (n_gt + n_hyp - n_matched)

    return precision


def matching_overlap(psg, fractions=(0.5,0.5)):
    """
    create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
    NOTE: a true matching is only gauranteed for fractions > 0.5. Otherwise some cells might have deg=2 or more.
    NOTE: doesnt break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
    """
    afrac, bfrac = fractions
    tmp = np.sum(psg+4e-6, axis=1, keepdims=True)

    m0 = np.where(tmp==0,0,psg / tmp)
    tmp = np.sum(psg, axis=0, keepdims=True)
    m1 = np.where(tmp==0,0,psg / tmp)
    m0 = m0 > afrac
    m1 = m1 > bfrac
    matching = m0 * m1
    matching = matching.astype('bool')
    return matching


def measure_seg(partial_dataset=False):
    def seg(lab_gt, lab, partial_dataset=partial_dataset):
        """
        calculate seg from pixel_sharing_bipartite
        seg is the average conditional-iou across ground truth cells
        conditional-iou gives zero if not in matching
        ----
        calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
        for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
        IoU as low as 1/2 that don't match, and thus have CIoU = 0.
        """
        psg = pixel_sharing_bipartite(lab_gt, lab)
        iou = intersection_over_union(psg)
        matching = matching_overlap(psg, fractions=(0.5, 0.))
        matching[0, :] = False
        matching[:, 0] = False
        n_gt = len(set(np.unique(lab_gt)) - {0})
        n_matched = iou[matching].sum()
        if partial_dataset:
            return n_matched, n_gt
        else:
            return n_matched / n_gt

    return seg


def isnotebook():
    """
    Checks if code is run in a notebook, which can be useful to determine what sort of progressbar to use.
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/24937408#24937408
    Returns
    -------
    bool
        True if running in notebook else False.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def compute_labels(prediction, threshold):
    prediction_fg = prediction[..., 1]
    pred_thresholded = prediction_fg > threshold
    labels, _ = ndimage.label(pred_thresholded)
    return labels

def seg(lab_gt, lab,eps=1e-4):
        """
        calculate seg from pixel_sharing_bipartite
        seg is the average conditional-iou across ground truth cells
        conditional-iou gives zero if not in matching
        ----
        calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
        for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
        IoU as low as 1/2 that don't match, and thus have CIoU = 0.
        """
        psg = pixel_sharing_bipartite(lab_gt, lab)
        iou = intersection_over_union(psg)
        matching = matching_overlap(psg, fractions=(0.5, 0.))
        matching[0, :] = False
        matching[:, 0] = False
        n_gt = len(set(np.unique(lab_gt)) - {0})
        n_matched = iou[matching].sum()+eps
        if np.isnan(n_matched):
            n_matched=eps

        seg= n_matched / (n_gt+eps)

        return seg

def precision(lab_gt, lab, iou=0.5, partial_dataset=False,eps=1e-4):
        """
        precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching
        :Authors:
            Coleman Broaddus
        """
        psg = pixel_sharing_bipartite(lab_gt, lab)
        matching = matching_iou(psg, fraction=iou)
        assert matching.sum(0).max() < 2
        assert matching.sum(1).max() < 2
        n_gt = len(set(np.unique(lab_gt)) - {0})
        n_hyp = len(set(np.unique(lab)) - {0})
        n_matched = matching.sum()+eps
        if partial_dataset:
            return n_matched, (n_gt + n_hyp - n_matched)
        else:
            return n_matched / (n_gt + n_hyp - n_matched+eps)

        return precision

def threshold_optimization(img,lbl,model,seg_weight=2):
  optimal=[]
  thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95]
  for x in thresholds:
    t_seg=[]
    t_prec=[]
    prediction = model.predict(img);

    for i in range(len(lbl)):
      image=prediction[i,:,:,:];
      label= compute_labels(image, x);
      t_seg.append(seg(lbl[i].astype(int)[:,:],label[:,:]));
      t_prec.append(precision(lbl[i].astype(int)[:,:],label[:,:],iou=0.5));
    optimal.append(seg_weight*np.nanmean(t_seg)+np.nanmean(t_prec))
  opt_threshold=thresholds[np.argmax(optimal)]
  return opt_threshold
# Network definitions

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Add, concatenate, Lambda
from tensorflow_examples.models.pix2pix import pix2pix

def MobileNetEncoder(input_size = (None,None,1),
         train_encoder=False,
         random_encoder_weights=True,
         output_channels=1,max_pooling=True,pre_load_weights=False,pretrained_model=None,Denoising=False):
  """
  Create an encoder based in MobileNet attached to a general decoder for segmentation
       Args:
            input_size (array of 3 int): dimensions of the input image.
            random_encoder_weights(bool,optional):whether to initialise the encoder's weights
               to random weights or the pretrained in the imagenet or to load previously trained ones
            Output_channels(int,optional):define the kind of segmentation(semantic)
            and number of elements to segmentate
            max_pooling(boolean,optional):whether to apply a max_pooling or average pooling
            pre_load_weights(boolean,optional): if we want to add to our model some pretrained weights for the previous layers
            pretrained_model:model that is going to act as the starting point for our new model
       Returns:
            model (Keras model): model containing the segmentation net created.
  """

    #Now we load the base MobileNetV2 architecture for the decoder
  if random_encoder_weights==False:
                  input_size=(None,None,3)
  encoder_model = tf.keras.applications.MobileNetV2(input_shape=input_size, include_top=False,
                                                    weights=None if random_encoder_weights else 'imagenet',
                                                    pooling='max'if max_pooling else 'avg')

    # Use the activations of these layers as the skip connections(blocks 1-13) and bottleneck(block 16)
  layer_names = [
     'block_1_expand_relu',
     'block_3_expand_relu',
     'block_6_expand_relu',
     'block_13_expand_relu',
     'block_16_project',
  ]
    #Now we select the previous layers
  layers = [encoder_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=encoder_model.input, outputs=layers)
    #Here we define the number of layers for the decoder
    # The function applies a convolution to recreate the image
  up_stack = [
    pix2pix.upsample(512, 3),  # 8x8 -> 16x16
    pix2pix.upsample(256, 3),  # 16x16 -> 32x32
    pix2pix.upsample(128, 3),  # 32x32 -> 64x64
    pix2pix.upsample(64, 3),   # 64x64 -> 128x128
  ]
# we set the whole encoder to be trainable or not
  down_stack.trainable = train_encoder
  encoder_model.trainable=train_encoder

  inputs = tf.keras.layers.Input(shape=input_size)
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model can be meant for denoising or classification
  if Denoising:
      last_denoising = tf.keras.layers.Conv2DTranspose(
            1, 3, strides=2,
            padding='same',activation='sigmoid')  #128x128 -> 256x256
      x = last_denoising(x)
  else:
    if  output_channels==1:
        last_denoising = tf.keras.layers.Conv2DTranspose(
            1, 3, strides=2,
            padding='same',activation='sigmoid')  #128x128 -> 256x256
        x = last_denoising(x)


    else:
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same',activation='softmax')  #128x128 -> 256x256
        x = last(x)

  model= tf.keras.Model(inputs=inputs, outputs=x)#Recreates a model setting the specific layer(softmax or sigmoid act function)
  model.trainable=True
  if pre_load_weights:
    #Loading weights layer by layer except from the last layer whose structure would change
    model.load_weights(pretrained_model)
    #for i in range((len(model.layers)-1)):
     #   model.get_layer(index=i).set_weights(pretrained_model.get_layer(index=i).get_weights())
      #  print('Loaded pre-trained weights from layer',i,'of',len(model.layers))


  return model

  # Regular U-Net

def UNet(input_size = (None,None,1),
         filters=16,
         activation='elu',
         kernel_initializer = 'he_normal',
         dropout_value=0.2,
         average_pooling=True,
         spatial_dropout=False,num_outputs=1,pre_load_weights=False,pretrained_model=None,train_encoder=True,train_bottleneck=False,train_decoder=True,denoising=False,skip_connection_training=True):
  """
  Create a U-Net for segmentation
       Args:
            input_size (array of 3 int): dimensions of the input image.
            filters (int, optional): number of channels at the first level of U-Net
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel
                initializer type.
            dropout_value (real value/list/None, optional): dropout value of each
                level and the bottleneck
            average_pooling (bool, optional): use average-pooling between U-Net
                levels (otherwise use max pooling).
            spatial_dropout (bool, optional): use SpatialDroput2D, otherwise regular Dropout
            train_encoder(bool): set to true if not specified, whether to train the encoder or to freeze its weights.
       Returns:
            model (Keras model): model containing the ResUNet created.
  """
  # make a list of dropout values if needed
  if type( dropout_value ) is float:
            dropout_value = [dropout_value]*5

  inputs = Input( input_size )
  # Encoder
  conv1 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(inputs)
  conv1 = SpatialDropout2D(dropout_value[0])(conv1) if spatial_dropout else Dropout(dropout_value[0],trainable=train_encoder) (conv1)
  conv1 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(conv1)
  pool1 = AveragePooling2D(pool_size=(2, 2))(conv1) if average_pooling else MaxPooling2D(pool_size=(2, 2),trainable=train_encoder)(conv1)

  conv2 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(pool1)
  conv2 = SpatialDropout2D(dropout_value[1])(conv2) if spatial_dropout else Dropout(dropout_value[1],trainable=train_encoder) (conv2)
  conv2 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(conv2)
  pool2 = AveragePooling2D(pool_size=(2, 2))(conv2) if average_pooling else MaxPooling2D(pool_size=(2, 2),trainable=train_encoder)(conv2)

  conv3 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(pool2)
  conv3 = SpatialDropout2D(dropout_value[2])(conv3) if spatial_dropout else Dropout(dropout_value[2],trainable=train_encoder) (conv3)
  conv3 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(conv3)
  pool3 = AveragePooling2D(pool_size=(2, 2))(conv3) if average_pooling else MaxPooling2D(pool_size=(2, 2),trainable=train_encoder)(conv3)

  conv4 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(pool3)
  conv4 = SpatialDropout2D(dropout_value[3])(conv4) if spatial_dropout else Dropout(dropout_value[3],trainable=train_encoder)(conv4)
  conv4 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_encoder)(conv4)
  pool4 = AveragePooling2D(pool_size=(2, 2))(conv4) if average_pooling else MaxPooling2D(pool_size=(2, 2),trainable=train_encoder)(conv4)

  # Bottleneck
  conv5 = Conv2D(filters*16, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_bottleneck)(pool4)
  conv5 = SpatialDropout2D(dropout_value[4])(conv5) if spatial_dropout else Dropout(dropout_value[4],trainable=train_bottleneck)(conv5)
  conv5 = Conv2D(filters*16, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_bottleneck)(conv5)

  # Decoder
  up6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same',trainable=train_decoder) (conv5)
  merge6 = concatenate([conv4,up6], axis = 3,trainable=skip_connection_training)
  conv6 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(merge6)
  conv6 = SpatialDropout2D(dropout_value[3])(conv6) if spatial_dropout else Dropout(dropout_value[3],trainable=train_decoder)(conv6)
  conv6 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(conv6)

  up7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same',trainable=train_decoder) (conv6)
  merge7 = concatenate([conv3,up7], axis = 3,trainable=skip_connection_training)
  conv7 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(merge7)
  conv7 = SpatialDropout2D(dropout_value[2])(conv7) if spatial_dropout else Dropout(dropout_value[2],trainable=train_decoder)(conv7)
  conv7 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(conv7)

  up8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same',trainable=train_decoder) (conv7)
  merge8 = concatenate([conv2,up8], axis = 3,trainable=skip_connection_training)
  conv8 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(merge8)
  conv8 = SpatialDropout2D(dropout_value[1])(conv8) if spatial_dropout else Dropout(dropout_value[1],trainable=train_decoder)(conv8)
  conv8 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(conv8)

  up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same',trainable=train_decoder) (conv8)
  merge9 = concatenate([conv1,up9], axis = 3,trainable=skip_connection_training)
  conv9 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(merge9)
  conv9 = SpatialDropout2D(dropout_value[0])(conv9) if spatial_dropout else Dropout(dropout_value[0],trainable=train_decoder)(conv9)
  conv9 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer,trainable=train_decoder)(conv9)
  if denoising:
      outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid') (conv9)
  else:
    if num_outputs==1:
            outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid') (conv9)
    else:
            outputs = Conv2D( num_outputs, (1, 1), activation='softmax') (conv9)


  model = Model(inputs=[inputs], outputs=[outputs])
  if pre_load_weights:
    #Loading weights layer by layer except from the last layer whose structure would change
      model.load_weights(pretrained_model)
      #for i in range((len(model.layers)-1)):
       # model.get_layer(index=i).set_weights(pretrained_model.get_layer(index=i).get_weights())
        #print('Loaded pre-trained weights from layer',i,'of',len(model.layers))
  if train_encoder==False:
      model.get_layer(index=0).trainable=False
#         for i in range(0,16):
#          model.get_layer(index=i).trainable=False
#         print('The encoder has been succesfully frozen')
#         if bottleneck_freezing:
#          model.get_layer(index=16).trainable=False
#          print('The bottleneck has been succesfully frozen')
  for layer in model.layers:
        print(layer, layer.trainable)

  return model


# == Residual U-Net ==

def residual_block(x, dim, filter_size, activation='elu',
                   kernel_initializer='he_normal', dropout_value=0.2, bn=False,
                   separable_conv=False, firstBlock=False, spatial_dropout=False):

    # Create shorcut
    shortcut = Conv2D(dim, activation=None, kernel_size=(1, 1),
                      strides=1)(x)

    # Main path
    if firstBlock == False:
        x = BatchNormalization()(x) if bn else x
        x = Activation( activation )(x)
    if separable_conv == False or firstBlock:
        x = Conv2D(dim, filter_size, strides=1, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, strides=1,
                            activation=None, kernel_initializer=kernel_initializer,
                            padding='same') (x)
    if dropout_value:
        x = SpatialDropout2D( dropout_value ) (x) if spatial_dropout else Dropout( dropout_value ) (x)
        print( str( dropout_value ) )
    x = BatchNormalization()(x) if bn else x
    x = Activation( activation )(x)

    if separable_conv == False:
        x = Conv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)

    # Add shortcut value to main path
    x = Add()([shortcut, x])
    print( 'residual block, dim: ' + str(dim) + ' , output shape: '+ str(x.shape) )
    return x

def level_block(x, depth, dim, fs, ac, k, d, bn, sc, fb, ap, spatial_dropout):
    do = d[depth] if d is not None else None
    if depth > 0:
        r = residual_block(x, dim, fs, ac, k, do, bn, sc, fb, spatial_dropout)
        x = AveragePooling2D((2, 2)) (r) if ap else MaxPooling2D((2, 2)) (r)
        x = level_block(x, depth-1, (dim*2), fs, ac, k, d, bn, sc, False, ap, spatial_dropout)
        x = Conv2DTranspose(dim, (2, 2), strides=(2, 2), padding='same') (x)
        x = Concatenate()([r, x])
        x = residual_block(x, dim, fs, ac, k, do, bn, sc, False, spatial_dropout)
    else:
        x = residual_block(x, dim, fs, ac, k, do, bn, sc, False, spatial_dropout)
    return x


def ResUNet( input_size=(None, None, 1), activation='elu', kernel_initializer='he_normal',
            dropout_value=0.2, batchnorm=False, average_pooling=False, separable=False,
            filters=16, depth=4, spatial_dropout=False, long_shortcut=True,num_outputs=1):

    """Create a Residual U-Net for segmentation
       Args:
            input_size (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel
            initializer type.
            dropout_value (real value/list/None, optional): dropout value of each
            level and the bottleneck
            batchnorm (bool, optional): use batch normalization
            average_pooling (bool, optional): use average-pooling between U-Net levels
            (otherwise use max pooling).
            separable (bool, optional): use SeparableConv2D instead of Conv2D
            filters (int, optional): number of channels at the first level of U-Net
            depth (int, optional): number of U-Net levels
            spatial_dropout (bool, optional): use SpatialDroput2D, otherwise regular Dropout
            long_shortcut (bool, optional): add long shorcut from input to output.
       Returns:
            model (Keras model): model containing the ResUNet created.
    """

    inputs = Input( input_size )
    if dropout_value is not None:
        if type( dropout_value ) is float:
            dropout_value = [dropout_value]*(depth+1)
        else:
            dropout_value.reverse() # reverse list to go from top to down

    x = level_block(inputs, depth, filters, 3, activation, kernel_initializer,
                    dropout_value, batchnorm, separable, True, average_pooling,
                    spatial_dropout)

    if long_shortcut:
        x = Add()([inputs,x]) # long shortcut
    if num_outputs==1:
         outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid' ) (x)
    else:
         outputs = Conv2D( num_outputs, (1, 1), activation='softmax' ) (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
import datetime
import statistics
from skimage import morphology
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border

def morphology_analysis(data,input,Source,Target):


    if Source!=Target:

            if Source=='Lucchi++':
                desired_ratio=0.004600387559051737

                if Target=='Kasthuri++':
                    delta=0.028086438236661453
                elif Target=='VNC':
                    delta=0
            elif Source=='Kasthuri++':
                desired_ratio=0.004182411757335

                if Target=='Lucchi++':
                    delta=0
                elif Target=='VNC':
                    delta=0
            elif Source=='VNC':
                desired_ratio=0.006005852143346416

                if Target=='Lucchi++':
                    delta=0
                elif Target=='Kasthuri++':
                    delta=0.028086438236661453
    p_area = []
    p_solidity = []
    p_eccentricity = []
    p_orientation = []
    n_objects=[]
    ratio_objects_area=[]
    #small_objs = 0 # Definir valor en función del GT de train. El remove se hace en 3D. De Lucchi++ a Kasthuri++ este valor lo he puesto en 4000

    label_img = label(data)   # Connected components
    #label_img = morphology.remove_small_objects(label_img, small_objs)  # Filtrar objetos pequeños

    factor=1
    for i in range(label_img.shape[0]):     # Por cada imagen 2D

        img = label_img[i]
        if Target=='Kasthuri++':
            area=(label_img.shape[1] * label_img.shape[2]-np.sum(np.sum((input[i]==0))))
        else:
            area=label_img.shape[1] * label_img.shape[2]



        props = regionprops_table(img, properties=('area', 'solidity', 'eccentricity', 'orientation'))    # Sacar las propiedades

        for v in props['area']:

            ratio=v/area*factor
            if v>10:
                p_area.append(v)
                ratio_objects_area.append(ratio)
        for v in props['solidity']: p_solidity.append(v)
        for v in props['eccentricity']: p_eccentricity.append(v)
        for v in props['orientation']: p_orientation.append(v)
        n_objects.append(len(np.unique(img))-1)
        factor+=delta


    try:
        gt_area_value = statistics.mean(p_area)
        gt_solidity_value = statistics.mean(p_solidity)
        gt_eccentricity_value = statistics.mean(p_eccentricity)
        gt_orientation_value = statistics.mean(p_orientation)
        gt_area_value_median=statistics.median(p_area)
        gt_object_number=statistics.mean(n_objects)
        gt_ratio=statistics.mean(ratio_objects_area)

    except:
        gt_area_value = 0
        gt_solidity_value = 0
        gt_eccentricity_value = 0
        gt_orientation_value = 0
        gt_area_value_median=0
        gt_object_number=0
        gt_ratio=0

    return gt_area_value,gt_solidity_value,gt_eccentricity_value,gt_orientation_value,gt_area_value_median,gt_object_number,gt_ratio

class CustomSaver(keras.callbacks.Callback):


    def __init__(self, X_test,Y_test,path_save='Models',Source=None,Target=None):
        import pandas as pd

        """ Save params in constructor
        """
        if Source!=Target:

            if Source=='Lucchi++':
                self.desired_ratio=0.004600387559051737

            elif Source=='Kasthuri++':
                self.desired_ratio=0.004182411757335

            elif Source=='VNC':
                self.desired_ratio=0.006005852143346416


        self.batch_size=1
        self.path_save=path_save
        self.Xtest=X_test
        self.Ytest=Y_test
        self.Xtest=np.asarray(X_test)
        self.Ytest = np.asarray(Y_test)
        self.Ytest = np.expand_dims( self.Ytest, axis=-1 )
        self.IoU_test=[]
        self.x=[]

        self.area=[]
        self.solidity=[]
        self.eccentricity=[]
        self.orientation=[]
        self.median_area=[]
        self.n_objects=[]
        self.ratio=[]
        self.update_count=0
        self.tol=0.05
        self.past_ratio=1
        self.Source=Source
        self.Target=Target



    def on_epoch_end(self, epoch, logs={}):

        if (epoch%2)==0 or epoch==0 or epoch==1:  # or save after some epoch, each k-th epoch etc.
            IoU_Dataset12Dataset1_temp=[]
            for i in range(0,len(self.Xtest)):

                normalizedImg = self.Xtest[i][:,:,:];
                prediction = self.model.predict(normalizedImg[np.newaxis,:,:]);
                image=prediction[0,:,:,0];
                filtered_img=((normalizedImg[:,:]!=0))
                filtered_img=image[:,:]*filtered_img[:,:,0]
                IoU_Dataset12Dataset1_temp.append(jaccard_index_final(self.Ytest[i][:,:,0],filtered_img[:,:]));

            jaccard=np.mean(np.nan_to_num(IoU_Dataset12Dataset1_temp))
            print('Jaccard in target: '+ str(jaccard))
            self.jaccard=jaccard

            self.IoU_test.append(jaccard)
            self.x.append(int(epoch))
            predictions = self.model.predict(self.Xtest,batch_size=1)
            print(predictions.shape)

            gt_area_value,gt_solidity_value,gt_eccentricity_value,gt_orientation_value,gt_area_value_median,gt_object_number,gt_ratio=morphology_analysis(predictions[:,:,:,0]>=0.5,self.Xtest,self.Source,self.Target)

            self.area.append(gt_area_value)
            self.solidity.append(gt_solidity_value)
            self.eccentricity.append(gt_eccentricity_value)
            self.orientation.append(gt_orientation_value)
            self.median_area.append(gt_area_value_median)
            self.n_objects.append(gt_object_number)
            self.ratio.append(gt_ratio)
            print('Ratio in target: '+ str(gt_ratio))
            print('Ratio in source: '+str(self.desired_ratio))

            # if epoch>=10:
            #     if  abs(gt_ratio-self.past_ratio)>=self.tol*gt_ratio:
            #         #Update if the difference between the actual and past ratio is  bigger than tol*actual_ratio
            #         self.update_count=0
            #         self.best_model=f'{self.path_save}model_E{epoch}_jaccard_{jaccard:.3f}.h5'
            #         self.top_epoch=epoch
            #     else:
            #         #Ratio hasn't changed more than 5e-4 for 2 updates(i.e 4 epochs)
            #         self.update_count+=1
            #         if self.update_count>5:
            #             #Stop the training as it may have converged to a value
            #             print("Training ratio is stable, so stopping training!!")
            #             self.best_model=f'{self.path_save}model_E{epoch}_jaccard_{jaccard:.3f}.h5'
            #             self.top_epoch=epoch
            #             self.model.stop_training = True
            # self.past_ratio=gt_ratio
            self.epoch=epoch


            if epoch>=10 or epoch<=1:
                if  abs(gt_ratio-self.desired_ratio)<=abs(self.past_ratio-self.desired_ratio):
                    #Update if the difference between the actual and past ratio is  bigger than tol*actual_ratio

                    self.best_model=f'{self.path_save}/model_ARAModel_jaccard_{jaccard:.3f}.h5'
                    self.top_epoch=epoch
                    self.past_ratio=gt_ratio

                    self.model.save_weights(self.best_model)
            print(f'model_E{epoch}_jaccard_{jaccard:.3f}_ratio_{gt_ratio:.3f}')

    def on_train_end(self,logs={}):
        #Save last epoch
        self.model.save_weights(f'{self.path_save}/model_LastEpoch_jaccard_{self.jaccard:.3f}_ratio_{self.past_ratio:.3f}.h5')
        #Load optimal model
        self.model.load_weights(self.best_model)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True,dpi=200)

        ax1.plot(self.x,self.IoU_test,color='black',marker='.',label='Target IoU');
        ax1.plot(self.top_epoch, self.IoU_test[int(self.x.index(self.top_epoch))], "ro",label='Optimal ratio model')


        ax1.set_ylabel('IoU')


        ax1.axhline(y=np.max(self.IoU_test), color='green', linestyle='dashed',label='Optimal IoU')
        ax1.legend()



        ax2.plot(self.x,self.ratio,color='black',marker='.',label='Target Ratio');

        ax2.set_xlabel('Number of epochs')
        ax2.set_ylabel('Ratio')
        ax2.set_yscale('log')

        #ax2.axhline(y=8.5e-3, color='green', linestyle='dashed',label='Goal Ratio')
        ax2.plot(self.top_epoch, self.ratio[int(self.x.index(self.top_epoch))], "ro",label='Optimal ratio model')
        ax2.legend()
        fig.suptitle('Target segmentation during Fine-tuning')
        plt.savefig('Target_evolution{}.png'.format(datetime.datetime.now().time()))
        plt.close()

        morphology=pd.DataFrame()
        morphology['Epochs']=self.x
        morphology['IoU']=self.IoU_test

        morphology['area']=self.area
        morphology['solidity']=self.solidity
        morphology['eccentricity']=self.eccentricity
        morphology['orientation']=self.orientation
        morphology['Median area']=self.median_area
        morphology['Object Number']=self.n_objects
        morphology['Ratio']=self.ratio

        morphology.to_csv('per_epoch_evolution{}.txt'.format(datetime.datetime.now().time()))





import tensorflow as tf
import numpy as np

def train(X_train,Y_train,X_val,Y_val,numEpochs,output_channels,patience,lr,min_lr,batch_size_value,schedule,model_name,optimizer_name,loss_acronym,max_pooling,train_encoder=True,train_decoder=True,random_encoder_weights=True,preTrain=False,Denoising=False,pre_load_weights=False,pretrained_model=None,plot_history=False,seg_weights=[1.,1.,5.],bottleneck_freezing=False,save_best_only=True,check_ev=False,path_save='saved_models',X_test=None,Y_test=None,Source=None,Target=None):
  """Inputs:
        numEpochs(int):number of "loops" of training
        patience(int): number of "loops" without improvement till the training is stopped, in the case of reduce till the lr is reduced to its half
        lr(float): number indicating the lr starting value// in the case of oneCycle the max lr
        batch_size_value(int):number of images in each step of training inside an epoch
        schedule(string):indicating the variations performed in the lr during the training #'oneCycle' # 'reduce' # None
        model_name(string):indicating the architecture to be used #'UNet','ResUNet','MobileNetEncoder'
        loss_acronym(string): indicating the name of the loss function to be applied 'BCE', 'Dice', 'W_BCE_Dice','CCE','SEG','mae'
        optimizer_name(string):indicating the kind of optimized to be used 'Adam', 'SGD'
        max_pooling(boolean):indicating True if max_pooling must be performed, False if average pooling has to be performed
        preTrain(boolean):indicating whether we're preTraining with denoising the network or training it for the final task
        train_encoder(boolean):indicating whether to freeze or not the training of the encoder part of the model in the case of MobileNetEncoder
        random_encoder_weights(boolean):indicating whether to initialize the model with random_gaussian weights or to use MobileNet imagenet pretrained weights//only available for MobileNet encoder
        Denoising(boolean):whether to tune the architecture of the network(by varying its last layer to be able to deal with denoising)
        pre_load_weights(boolean):whether to start by loading some weights from another model
        pretrained_model:keras model object from where the weigths must be extracted
        plot_history(boolean): indicating whether to plot the train and validation loss graphs


      Output:
      history: containing the training of the model
      model: keras model trained for a particular task
  """
  bottleneck_train=True
  tf.keras.backend.clear_session()
  if bottleneck_freezing:
    bottleneck_train=False
  #Here we create the training and validation generators
  # define data generators to do data augmentation
  train_generator, val_generator = get_train_val_generators( X_train,
                                                          Y_train,
                                                         X_val,Y_val,
                                                          rescale= None,
                                                          horizontal_flip=True,
                                                          vertical_flip=True,
                                                          rotation_range = 180,
                                                          #width_shift_range=0.2,
                                                          #height_shift_range=0.2,
                                                          #shear_range=0.2,
                                                          preprocessing_function=None,
                                                          batch_size=batch_size_value,
                                                          show_examples=False )
  #Here we establish the architecture based in the input model_name

  num_filters=16
  dropout_value=0.2
  if model_name == 'UNet':
      model = UNet( filters=num_filters, dropout_value=dropout_value,
                   spatial_dropout=False, average_pooling=False, activation='elu',num_outputs=output_channels,pre_load_weights=pre_load_weights,pretrained_model=pretrained_model,train_encoder=train_encoder,train_bottleneck=bottleneck_train,train_decoder=train_decoder,denoising=Denoising)
  elif model_name == 'ResUNet':
      model = ResUNet( filters=num_filters, batchnorm=False, spatial_dropout=True,
                      average_pooling=False, activation='elu', separable=False,
                      dropout_value=dropout_value,num_outputs=output_channels )
  elif model_name=='MobileNetEncoder':
      model =MobileNetEncoder(
           train_encoder=train_encoder,
           random_encoder_weights=random_encoder_weights,
           output_channels=output_channels,
          max_pooling=max_pooling,pre_load_weights=pre_load_weights,pretrained_model=pretrained_model,Denoising=Denoising)
  elif model_name=='AttentionUNET':
    model=Attention_U_Net_2D(image_shape = (None,None,1), activation='elu', feature_maps=[16, 32, 64, 128, 256],
                       drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, batch_norm=False,
                       k_init='he_normal',num_outputs=1,pre_load_weights=pre_load_weights,pretrained_model=pretrained_model,train_encoder=train_encoder,bottleneck_train=bottleneck_train,skip_connection_train=True,denoising=Denoising,train_decoder=train_decoder)

  model.summary()

  if optimizer_name == 'SGD':
      optim =  tf.keras.optimizers.SGD(
              lr=lr, momentum=0.99, decay=0.0, nesterov=False)
  elif optimizer_name == 'Adam':
      optim = tf.keras.optimizers.Adam( learning_rate=lr )

  if loss_acronym == 'BCE':
      loss_funct = 'binary_crossentropy'
  elif loss_acronym == 'Dice':
      loss_funct = dice_loss
  elif loss_acronym == 'W_BCE_Dice':
      loss_funct = weighted_bce_dice_loss(w_bce=0.8, w_dice=0.2)
  elif loss_acronym== 'CCE':
      loss_funct= tf.keras.losses.CategoricalCrossentropy()
  elif loss_acronym=='mse':
      loss_funct='mse'
  elif loss_acronym=='mae':
      loss_funct='mean_absolute_error'
  elif loss_acronym=='SEG':
      loss_funct=loss_seg(relative_weights=seg_weights)

  if preTrain:
    eval_metric = 'mean_absolute_error'
    model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric])
  else:
    if loss_acronym == 'BCE':
      eval_metric = jaccard_index_final
    else:
       eval_metric = jaccard_index
    model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric])

  # compile the model with the specific optimizer, loss function and metric


    # callback for early stop
  earlystopper = EarlyStopping(patience=numEpochs, verbose=1, restore_best_weights=True)

  if schedule == 'oneCycle':
      # callback for one-cycle schedule
      steps = np.ceil(len(X_train) / batch_size_value) * numEpochs
      lr_schedule = OneCycleScheduler(lr, steps)
  elif schedule == 'reduce':
      # callback to reduce the learning rate in the plateau
     lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                               patience=patience, min_lr=min_lr)
  else:
      lr_schedule = None
  if schedule=='oneCycle':
     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=(path_save+'/best_model'),verbose=1,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False,save_freq='epoch')
  else:
      model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=(path_save+'/best_model'),verbose=1,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

  save_periodically= CustomSaver(X_test,Y_test,path_save,Source,Target)
  callbacks = [earlystopper,model_checkpoint_callback] if lr_schedule is None else [earlystopper, lr_schedule,model_checkpoint_callback]
  if check_ev:
      callbacks = [earlystopper,save_periodically] if lr_schedule is None else [earlystopper, lr_schedule,save_periodically]
  # train!
  validation_steps=np.ceil(len(X_val[:,0,0,0])/batch_size_value)
  steps_per_epoch=np.ceil(len(X_train[:,0,0,0])/batch_size_value)
  del X_train,Y_train,X_val,Y_val
  history = model.fit(train_generator, validation_data=val_generator,
                      validation_steps=validation_steps,
                      steps_per_epoch=steps_per_epoch,
                      epochs=numEpochs, callbacks=callbacks,verbose=2)
  if check_ev==False:
    print('Restoring model...')
    model.load_weights(filepath=(path_save+'/best_model'))
    print('Done!')

  if True:
    plt.figure(figsize=(14,5))

    if callable( eval_metric ):
     metric_name = eval_metric.__name__
    else:
      metric_name = eval_metric

    # summarize history for loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

     # summarize history for metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history[metric_name])
    plt.plot(history.history['val_'+metric_name])
    plt.title('model ' + metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('Train_Val_evolution{}.png'.format(datetime.datetime.now().time()))

  return history,model
import numpy as np
import os
import random
import tensorflow as tf
import cv2
from skimage.util import img_as_ubyte
from skimage import io,color
import matplotlib.pyplot as plt

def create_patches( imgs,lbls,patch_size,add_noise=False,noise_level=0,random_patches=False,factor=1,filter=False,threshold=0.02,verbose=False):
    ''' Create a list of  patches out of a list of images
    Args:
        imgs: list of input images
        patch_size:list including both dimensions (256,256)
        add_noise: boolean to add noise to the cropped image(useful for denoising previous steps or superresolution)
        noise_level: int between 0-255 representing the sd of the gaussian noise added

    ¡¡¡¡¡IMPORTANT if the image is not in a greyscale of 0-255 the noise must be rescaled in between 0-1 !!!!
        percentage_data:0-1 float specifying the percentage of data used for training
    Returns:
        list of image patches
    '''

    if random_patches:
        print('Randomly cropping patches from the original image')
        lbl_patches=[]
        patches = [] #empty list to store the corresponding patches
        patch_height=patch_size[0]
        patch_width=patch_size[1]
        for n in range( 0, len( imgs ) ):
            img = imgs[ n ]
            lbl=lbls[n]
            original_size = imgs[n].shape
            num_y_patches = original_size[ 0 ] // patch_size[0]#obtain the int number of patches that can be actually extracted from the original image
            num_x_patches = original_size[ 1 ] // patch_size[1]
            num=num_y_patches*num_x_patches*factor
            w=0
            while w<int(num):

                i=random.choice(range( 0, num_y_patches ))
                j=random.choice(range( 0, num_x_patches ))
                patch_lbl=lbl[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ]
                if filter:
                    if np.mean(np.mean(patch_lbl))>=threshold:

                        patches.append(img[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ])
                        lbl_patches.append(lbl[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ])
                        w+=1
                        if verbose:print(num)
                        if verbose:print(w)
                    else:
                      if verbose:print('Non-significative patch')
                      if verbose:print(np.mean(np.mean(patch_lbl)))
                else:
                    patches.append(img[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ])
                    lbl_patches.append(lbl[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ])

                    w+=1

    else:
        print('Sequentially cropping patches from the original image')
        lbl_patches=[]
        patches = [] #empty list to store the corresponding patches
        patch_height=patch_size[0]
        patch_width=patch_size[1]
        for n in range( 0, len( imgs ) ):
            image = imgs[ n ]
            lbl=lbls[n]
            original_size = imgs[n].shape
            num_y_patches = original_size[ 0 ] // patch_size[0]#obtain the int number of patches that can be actually extracted from the original image
            num_x_patches = original_size[ 1 ] // patch_size[1]
            for i in range( 0, num_y_patches ):
                for j in range( 0, num_x_patches ):
                    if add_noise:
                        trainNoise = np.random.normal(loc=0, scale=noise_level, size=(patch_width,patch_height))
                        patches.append(np.clip(image[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ]+trainNoise,0,255)  )
                        lbl_patches.append(lbl[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ]  )
                    else:
                        patches.append(image[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ]  )
                        lbl_patches.append(lbl[ i * patch_width : (i+1) * patch_width,
                                            j * patch_height : (j+1) * patch_height ]  )

    return patches,lbl_patches

def filter_patches(patch,gt_patch,percent):
    '''
    select_percent: float 0-1 representing the number of positive pixels in a patch to accept it as informative


    '''
    if len(patch)==len(gt_patch):
        print('Number of patches and labels is equal')
    else:
        return print('Error different number of patches')
    preserved_patches=[]
    preserved_GT=[]
    for i in range(0,len(patch)):
        if np.mean(np.mean(gt_patch[i]))>=percent:
            preserved_patches.append(patch[i])
            preserved_GT.append(gt_patch[i])
    print('Deleted:'+str(len(patch)-len(preserved_patches))+' non-informative patches')
    print('Conserved:'+str(len(preserved_patches)))
    return preserved_patches, preserved_GT


def set_seed(seedValue=42):
  """Sets the seed on multiple python modules to obtain results as
  reproducible as possible.
  Args:
  seedValue (int, optional): seed value.
  """
  random.seed(a=seedValue)
  np.random.seed(seed=seedValue)
  tf.random.set_seed(seedValue)
  os.environ["PYTHONHASHSEED"]=str(seedValue)

def shuffle_fragments( imgs,number_of_patches=(3,3)):
    ''' Shuffles different fragments of the input imgs
    Args:
        imgs: list of input images
        number_of_patches: (x,y) containing the number of divisions per x and per y
    Returns:
        list of image patches
    '''
    patches=[]

    original_size = imgs.shape
    img=1*imgs# This multiplication is made to avoid further relating both variables
    num_y_patches = number_of_patches[1]#obtain the int number of patches that can be actually extracted from the original image
    num_x_patches = number_of_patches[0]
    patch_height=original_size[0]//num_x_patches
    patch_width=original_size[1]//num_y_patches
    for i in range( 0, num_y_patches ):
                for j in range( 0, num_x_patches ):

                    patches.append(img[ i * patch_width : (i+1) * patch_width,
                                          j * patch_height : (j+1) * patch_height ]  )
    k=0
    random.shuffle(patches)
    for i in range( 0, num_y_patches ):
            for j in range( 0, num_x_patches ):

                img[ i * patch_width : (i+1) * patch_width,
                                          j * patch_height : (j+1) * patch_height ]=patches[k]
                k+=1
    return img

def hide_fragments( imgs,patch_size,percent):
    ''' Sets to 0 different fragments of the input imgs
    Args:
        imgs: list of input images
        patch_size: list including both dimensions of the fragment to  (256,256)
        percent: representing the percentage of the total image to set to 0
    Returns:
        list of image patches
    '''
    patch_height=patch_size[0]
    patch_width=patch_size[1]
    original_size = imgs.shape
    img=1*imgs# This multiplication is made to avoid further relating both variables
    num_y_patches = original_size[ 0 ] // patch_size[0]#obtain the int number of patches that can be actually extracted from the original image
    num_x_patches = original_size[ 1 ] // patch_size[1]
    n=percent*num_y_patches*num_x_patches

    for w in range(0,int(n)):
        i=random.choice(range( 0, num_y_patches ))
        j=random.choice(range( 0, num_x_patches ))
        img[ i * patch_width : (i+1) * patch_width,
                                  j * patch_height : (j+1) * patch_height ]=0

    return img


def add_Gaussian_Noise(image,percentage_of_noise,print_img=False):
  """
  image:  image to be added Gaussian Noise with 0 mean and a certain std
  percentage_of_noise:similar to 1/SNR, it represents the % of
  the maximum value of the image that will be used as the std of the Gaussian Noise distribution
  """
  max_value=np.max(image)
  noise_level=percentage_of_noise*max_value
  Noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
  noisy_img=np.clip(image+Noise,0,max_value)
  if print_img:
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow( image, 'gray' )
    plt.title( 'Original image' );
    # and its "ground truth"
    plt.subplot(1, 2, 2)
    plt.imshow( noisy_img, 'gray' )
    plt.title( 'Noisy image' );

  return noisy_img

def crappify(img,resizing_factor,add_noise=True,noise_level=None,Down_up=True):

  """
  img: img to be modified
  resizing_factor(float): downsizing factor to divide the number of pixels with
  add_noise(boolean): indicating whether to add gaussian noise before applying the resizing
  noise_level(float): number between ]0,1] indicating the std of the Gaussian noise N(0,std)
  Down_up(boolean): indicating whether to perform a final upsampling operation
  to obtain an image of the same size as the original but with the corresponding loss of quality of downsizing and upsizing
  """
  w,h=img.shape
  org_sz=(h,w)
  new_w=int(w/np.sqrt(resizing_factor))
  new_h=int(h/np.sqrt(resizing_factor))
  targ_sz=(new_h,new_w)
  #add Gaussian noise
  if add_noise:
    noisy=add_Gaussian_Noise(img,noise_level,print_img=False)
    #downsize_resolution
    resized = cv2.resize(noisy, targ_sz, interpolation = cv2.INTER_LINEAR)
    #upsize_resolution
    if Down_up:
      resized=cv2.resize(resized, org_sz, interpolation = cv2.INTER_LINEAR)
  else:
    #downsize_resolution
    resized = cv2.resize(img, targ_sz, interpolation = cv2.INTER_LINEAR)
    #upsize_resolution
    if Down_up:
      resized=cv2.resize(resized, org_sz, interpolation = cv2.INTER_LINEAR)

  return resized

def reduce_number_imgs(imgs,label_imgs,percentage_data=1,normalize=True,imagenet=False):
    """
    Input:
    imgs:a list or tensor containing several images to be packed as a list after reducing its number
    label_imgs: a list or tensor containing several label images in the same order as the imgs tensor
    percentage_data: float(0-1) indicating the reduction in labels to be performed i.e 1 means that all the image will be taken into account
    normalize: Boolean indicating whether or not to perform a normalization step in the img, no normalization is performed in the labels as it is supposed that they would already been in a binary
    Output:
    x: list containing a subset of imgs
    y:list containing a subset of labels
    """
    n=len(imgs)
    if imagenet:
      if normalize:

        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [cv2.normalize(imgs[i]/np.max(imgs[i]), None, 0, 1, cv2.NORM_MINMAX) for i in idx]
        y= [label_imgs[i] for i in idx]
      else:
        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [color.gray2rgb(imgs[i]) for i in idx]
        y= [label_imgs[i] for i in idx]
    else:
      if normalize:

        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [cv2.normalize(imgs[i]/np.max(imgs[i]), None, 0, 1, cv2.NORM_MINMAX) for i in idx]
        y= [label_imgs[i] for i in idx]
      else:
        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [imgs[i] for i in idx]
        y= [label_imgs[i] for i in idx]
    print('Created list with '+str(len(x))+' images')

    return x,y

def append_blackborder(img,height,width):
  """ Function to append a blackborder to the images in order to avoid a resizing step that may affect the resolution and pixel size
  """
  new_h=(height-img.shape[0])
  new_w=(width- img.shape[1])
  img = cv2.copyMakeBorder(img ,new_h,0,new_w,0 , cv2.BORDER_CONSTANT)
  return img

def append_pot2(img):
  """
  Function to append a blackborder but instead of having to specify the shape of the desired image
  the function would check the shape and append a black border in order to obtain an image that is a multiple of 2^n as required by the U-Net Models

  """
  new_height=img.shape[0]
  new_width=img.shape[1]
  while new_height%32!=0:
    new_height+=1
  while new_width%32!=0:
    new_width+=1
  img = append_blackborder(img,new_height,new_width)
  #print('An image with shape'+str(img.shape)+'has been created')
  return img
import tensorflow as tf

def jaccard_index( y_true, y_pred, skip_first_mask=False ):
    ''' Define Jaccard index for multiple labels.
        Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
            skip_background (bool, optional): skip 0-label from calculation.
        Return:
            jac (tensor): Jaccard index value
    '''
    t=0.5
    if tf.shape(y_true)[-1]==1:
          y_pred_ = tf.cast(y_pred>t , dtype=tf.int32)
          y_true = tf.cast(y_true, dtype=tf.int32)

          TP = tf.math.count_nonzero(y_pred_ * y_true)
          FP = tf.math.count_nonzero(y_pred_ * (y_true - 1))
          FN = tf.math.count_nonzero((y_pred_ - 1) * y_true)

          jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: tf.cast(0.000, dtype='float64'))
    else:
        # We read the number of classes from the last dimension of the true labels
        num_classes = tf.shape(y_true)[-1]
        # One_hot representation of predicted segmentation after argmax
        y_pred_ = tf.one_hot(tf.math.argmax(y_pred, axis=-1), num_classes)
        y_pred_ = tf.cast(y_pred_, dtype=tf.int32)
        # y_true is already one-hot encoded
        y_true_ = tf.cast(y_true, dtype=tf.int32)
        # Skip background pixels from the Jaccard index calculation
        if skip_first_mask:
          y_true_ = y_true_[...,1:]
          y_pred_ = y_pred_[...,1:]

        TP = tf.math.count_nonzero(y_pred_ * y_true_)
        FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
        FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

        jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                      lambda: tf.cast(0.000, dtype='float64'))

    return jac

def jaccard_index_final(y_true, y_pred, t=0.5):
  """Define Jaccard index for final evaluation .
      Args:
          y_true (tensor): ground truth masks.
          y_pred (tensor): predicted masks.
          t (float, optional): threshold to be applied.
      Return:
          jac (tensor): Jaccard index value
      additional: this metric is meant to output the same as the jaccard_index above but only for a single mask
  """

  y_pred_ = tf.cast(y_pred>t , dtype=tf.int32)
  y_true = tf.cast(y_true>0, dtype=tf.int32)

  TP = tf.math.count_nonzero(y_pred_ * y_true)
  FP = tf.math.count_nonzero(y_pred_ * (y_true - 1))
  FN = tf.math.count_nonzero((y_pred_ - 1) * y_true)

  jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                lambda: tf.cast(0.000, dtype='float64'))

  return jac

from tensorflow.keras import losses

def dice_coeff(y_true, y_pred):
    """Define Dice coefficient.
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
       Return:
            score (tensor): Dice coefficient value
    """
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

# Dice coefficient loss (1 - Dice coefficient)

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# Loss function combining binary cross entropy and Dice loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# Weighted BCE+Dice
# Inspired by https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0

def weighted_bce_dice_loss(w_dice=0.5, w_bce=0.5):
    def loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) * w_bce + dice_loss(y_true, y_pred) * w_dice
    return loss




import keras.backend as K
#Based in denoiseg loss function

def loss_seg(relative_weights=[1.0,1.0,5.0]):
    """
    It is based in the DenoiSeg training function used in their paper for segmentation
    Calculates Cross-Entropy Loss between the class targets and predicted outputs.
    Predicted outputs consist of three classes: Foreground, Background and Border.
    Class predictions are weighted by the parameter `relative_weights`.

    """

    class_weights = tf.constant([relative_weights])
    def seg_crossentropy(class_targets, y_pred):


        onehot_labels = tf.reshape(class_targets, [-1, 3])# maintains the 3 dimensions for the labels regardless the size of the image or the batch size
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)#performs a weighted sum over a particular dimension of the tensor

        a = tf.reduce_sum(onehot_labels, axis=-1)#performs once again a sum over a particular dimension

        loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                          logits=tf.reshape(y_pred, [-1, 3]))#computes the softmax cat crossentropy

        weighted_loss = loss * weights #obtains a loss weighted by the number of labels and samples (the more positive of a label the higher importance )

        return K.mean(a * weighted_loss)# weights once again the number of positive labels per class
    return seg_crossentropy

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One cycle policy based on Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf)
Created on Wed Mar 31 13:53:39 2021

"""
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.callbacks import Callback

class CosineAnnealer:

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    """ `Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)],
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dropout, SpatialDropout2D, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,
                                     ELU, BatchNormalization, Activation, ZeroPadding2D, multiply, Lambda, UpSampling2D,
                                     Add, Multiply)
def Attention_U_Net_2D(image_shape = (None,None,1), activation='elu', feature_maps=[16, 32, 64, 128, 256],
                       drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, batch_norm=False,
                       k_init='he_normal',num_outputs=1,pre_load_weights=False,pretrained_model=None,train_encoder=True,bottleneck_train=True,skip_connection_train=True,denoising=False,train_decoder=True):
    """Create 2D U-Net with Attention blocks.
       Based on `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
       Parameters
       ----------
       image_shape : 2D tuple
           Dimensions of the input image.
       activation : str, optional
           Keras available activation type.
       feature_maps : array of ints, optional
           Feature maps to use on each level.
       drop_values : float, optional
           Dropout value to be fixed. If no value is provided the default behaviour will be to select a piramidal value
           starting from ``0.1`` and reaching ``0.3`` value.
       spatial_dropout : bool, optional
           Use spatial dropout instead of the `normal` dropout.
       batch_norm : bool, optional
           Make batch normalization.
       k_init : string, optional
           Kernel initialization for convolutional layers.
       n_classes: int, optional
           Number of classes.
       Returns
       -------
       model : Keras model
           Model containing the Attention U-Net.
       Example
       -------
       Calling this function with its default parameters returns the following network:
       .. image:: ../img/unet.png
           :width: 100%
           :align: center
       Image created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
       That networks incorporates in skip connecions Attention Gates (AG), which
       can be seen as follows:
       .. image:: ../img/attention_gate.png
           :width: 100%
           :align: center
       Image extracted from `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.
    """

    if len(feature_maps) != len(drop_values):
        raise ValueError("'feature_maps' dimension must be equal 'drop_values' dimension")
    depth = len(feature_maps)-1

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    x = Input(dinamic_dim)
    #x = Input(image_shape)
    inputs = x

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=train_encoder) (x)
        x = BatchNormalization(trainable=train_encoder) (x) if batch_norm else x
        x = Activation(activation,trainable=train_encoder) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i],trainable=train_encoder) (x)
            else:
                x = Dropout(drop_values[i],trainable=train_encoder) (x)
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=train_encoder) (x)
        x = BatchNormalization(trainable=train_encoder) (x) if batch_norm else x
        x = Activation(activation,trainable=train_encoder) (x)

        l.append(x)

        x = MaxPooling2D((2, 2),trainable=train_encoder)(x)
    encoder_layers=[]
    j=0
    #for layer in x.layers:
     #     j+=1
      #    layer._name = 'encoder_layer_'+str(j)
       #   encoder_layers.append(layer._name)


    # BOTTLENECK
    x = Conv2D(feature_maps[depth], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=bottleneck_train)(x)
    x = BatchNormalization(trainable=bottleneck_train) (x) if batch_norm else x
    x = Activation(activation,trainable=bottleneck_train) (x)
    if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[depth],trainable=bottleneck_train) (x)
            else:
                x = Dropout(drop_values[depth],trainable=bottleneck_train) (x)
    x = Conv2D(feature_maps[depth], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=bottleneck_train) (x)
    x = BatchNormalization(trainable=bottleneck_train) (x) if batch_norm else x
    x = Activation(activation,trainable=bottleneck_train) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv2DTranspose(feature_maps[i], (2, 2), strides=(2, 2), padding='same',trainable=train_decoder) (x)
        attn = AttentionBlock(x, l[i], feature_maps[i], batch_norm,trainable=skip_connection_train)
        x = concatenate([x, attn])
        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=train_decoder) (x)
        x = BatchNormalization(trainable=train_decoder) (x) if batch_norm else x
        x = Activation(activation,trainable=train_decoder) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i],trainable=train_decoder) (x)
            else:
                x = Dropout(drop_values[i],trainable=train_decoder) (x)

        x = Conv2D(feature_maps[i], (3, 3), activation=None, kernel_initializer=k_init, padding='same',trainable=train_decoder) (x)
        x = BatchNormalization(trainable=train_decoder) (x) if batch_norm else x
        x = Activation(activation,trainable=train_decoder) (x)


    if denoising:
        outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid') (x)
    else:
        if num_outputs==1:
            outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid') (x)
        else:
            outputs = Conv2D( num_outputs, (1, 1), activation='softmax') (x)



    model = Model(inputs=[inputs], outputs=[outputs])
    if pre_load_weights:
        #Loading weights layer by layer except from the last layer whose structure would change
        model.load_weights(pretrained_model)
        #for i in range((len(model.layers)-1)):
         #   model.get_layer(index=i).set_weights(pretrained_model.get_layer(index=i).get_weights())
          #  print('Loaded pre-trained weights from layer',i,'of',len(model.layers))
    if train_encoder==False:
              #Now we select the previous layers
            #for name in encoder_layers:
              #model.get_layer(name).trainable=train_encoder

            #for i in range(0,25):
              model.get_layer(index=0).trainable=False
             # print('The encoder has been succesfully frozen')
    #if bottleneck_train==False:
              #for i in range(25,33):
              #  model.get_layer(index=i).trainable=False
               # print('The bottleneck has been succesfully frozen')

    for layer in model.layers:
          print(layer.name, layer.trainable)

    return model


def AttentionBlock(x, shortcut, filters, batch_norm,trainable=False):
    """Attention block.
       Extracted from `Kaggle <https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64367>`_.
       Parameters
       ----------
       x : Keras layer
           Input layer.
       shortcut : Keras layer
           Input skip connection.
       filters : int
           Feature maps to define on the Conv layers.
       batch_norm : bool, optional
           To use batch normalization.
       Returns
       -------
       out : Keras layer
           Last layer of the Attention block.
    """
    input=x
    g1 = Conv2D(filters, kernel_size = 1,trainable=trainable)(shortcut)
    g1 = BatchNormalization(trainable=trainable) (g1) if batch_norm else g1
    x1 = Conv2D(filters, kernel_size = 1,trainable=trainable)(x)
    x1 = BatchNormalization(trainable=trainable) (x1) if batch_norm else x1

    g1_x1 = Add(trainable=trainable)([g1,x1])
    psi = Activation('relu',trainable=trainable)(g1_x1)
    psi = Conv2D(1, kernel_size = 1,trainable=trainable)(psi)
    psi = BatchNormalization(trainable=trainable) (psi) if batch_norm else psi
    psi = Activation('sigmoid',trainable=trainable)(psi)
    x = Multiply(trainable=trainable)([x,psi])

    return x

def gpu_select(GPU_availability,GPU):

  if GPU_availability:
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
      os.environ["CUDA_VISIBLE_DEVICES"] = GPU;
  print('GPU:'+GPU+' was selected')

def set_seed(seedValue=42):
  """Sets the seed on multiple python modules to obtain results as
  reproducible as possible.
  Args:
  seedValue (int, optional): seed value.
  """
  random.seed(a=seedValue)
  np.random.seed(seed=seedValue)
  tf.random.set_seed(seedValue)
  os.environ["PYTHONHASHSEED"]=str(seedValue)

from skimage.util import img_as_ubyte
from skimage import io
from matplotlib import pyplot as plt

def load_img(train_input_path,train_label_path):
    #loading and sorting the filenames in order to read the images
  train_input_filenames = [x for x in os.listdir( train_input_path) if x.endswith(".png") or x.endswith('.tif')]
  train_input_filenames.sort()
  train_label_filenames = [x for x in os.listdir( train_label_path ) if x.endswith(".png") or x.endswith('.tif')]
  train_label_filenames.sort()
  print('Loading data from '+ train_input_path)
  print( 'Dataset input images loaded: ' + str( len(train_input_filenames)) )
  print( 'Dataset label images loaded: ' + str( len(train_label_filenames)) )

  # read training images and labels
  train_img = [ io.imread( train_input_path + '/' + x, as_gray=True ) /255.0 for x in train_input_filenames ]
  train_lbl = [ io.imread( train_label_path + '/' + x, as_gray=True ) /255.0 for x in train_label_filenames ]
  return train_img, train_lbl

def prepare_training_data(imgs,lbls):
  #  input
  X = np.asarray(imgs)
  X = np.expand_dims( X, axis=-1 ) # add extra dimension
  print(X[0].shape)

  #  ground truth
  Y = np.asarray(lbls)#here we define our ground_truth
  Y = np.expand_dims( Y, axis=-1 ) # add extra dimension
  print(Y[0].shape)
  return X,Y

def evaluate_ranges(X):
  #@title
  values=[]
  for i in range(len(X[:,0,0,0])):
    values.append(np.max(X[i,:,:,:]))
  print('The range of max values is between:',np.min(values),'and',np.max(values))

def prepare_test_data(test_img,test_lbl):

  X_test = [  np.expand_dims( append_pot2(x), axis=-1 )  for x in test_img ];
  Y_test = [  append_pot2(x)  for x in test_lbl ];
  test_lbl=[  append_pot2(x)  for x in test_lbl ];
  print(X_test[0].shape)
  print(Y_test[0].shape)

  return X_test,Y_test,test_lbl

from PIL import Image

def evaluate_test(X_test,test_lbl,model,save_img=False,path=None):
    IoU_Dataset12Dataset1_temp=[]
    for i in range(0,len(X_test)):

      print('Evaluating test image',i)
      normalizedImg = X_test[i][:,:,:];
      prediction = model.predict(normalizedImg[np.newaxis,:,:]);
      image=prediction[0,:,:,:];
      filtered_img=((normalizedImg[:,:]!=0))
      filtered_img=image[:,:,0]*filtered_img[:,:,0]
      if save_img:
        try:
            Image.fromarray((filtered_img[:,:]*255).astype(np.uint8)).save(f'{path}/prediction_{str(i)}.png')
        except Exception as e: print(e)

      IoU_Dataset12Dataset1_temp.append(jaccard_index_final(test_lbl[i],filtered_img));
    return np.mean(np.nan_to_num(IoU_Dataset12Dataset1_temp))


import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders


def send_mail(send_from, send_to, subject, message, files=[],
              server="localhost", port=587, username='', password='',
              use_tls=True):
    """Compose and send email with provided info and attachments.

    Args:
        send_from (str): from name
        send_to (list[str]): to name(s)
        subject (str): message title
        message (str): message body
        files (list[str]): list of file paths to be attached to email
        server (str): mail server host name
        port (int): port number
        username (str): server auth username
        password (str): server auth password
        use_tls (bool): use TLS mode
    """
    try:


        msg = MIMEMultipart()
        msg['From'] = send_from
        msg['To'] = COMMASPACE.join(send_to)
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject

        msg.attach(MIMEText(message))

        for path in files:
            part = MIMEBase('application', "octet-stream")
            with open(path, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                            'attachment; filename={}'.format(Path(path).name))
            msg.attach(part)

        smtp = smtplib.SMTP(server, port)
        if use_tls:
            smtp.starttls()
        smtp.login(username, password)
        smtp.sendmail(send_from, send_to, msg.as_string())
        smtp.quit()
        print ("Email sent successfully!")
    except Exception as ex:
        print ("Something went wrong….",ex)

def reduce_number_imgs_num(imgs,label_imgs,num_patches=1,normalize=True,imagenet=False):
    n=len(imgs)
    if imagenet:
      if normalize:

        idx=random.sample(list(range(0,n)),int(num_patches))
        x= [cv2.normalize(imgs[i]/np.max(imgs[i]), None, 0, 1, cv2.NORM_MINMAX) for i in idx]
        y= [label_imgs[i] for i in idx]
      else:
        idx=random.sample(list(range(0,n)),int(num_patches))
        x= [color.gray2rgb(imgs[i]) for i in idx]
        y= [label_imgs[i] for i in idx]
    else:
      if normalize:

        idx=random.sample(list(range(0,n)),int(num_patches))
        x= [cv2.normalize(imgs[i]/np.max(imgs[i]), None, 0, 1, cv2.NORM_MINMAX) for i in idx]
        y= [label_imgs[i] for i in idx]
      else:
        idx=random.sample(list(range(0,n)),int(num_patches))
        x= [imgs[i] for i in idx]
        y= [label_imgs[i] for i in idx]
    print('Created list with '+str(len(x))+' images')

    return x,y

from skimage.feature import hog

def hog_image(img,orientations=6,pixels_per_cell=(4,4),cells_per_block=(2,2)):
    fd, output = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                	cells_per_block=cells_per_block, visualize=True, multichannel=False)
    return output


def visualize_feature_maps(model,layer,normalize=True,n_filters=6,save_img=False,path=None):

	# model: this is a already keras' loaded model not a h5 file
	# layer: the layer whose weights we want to visualize
	# normalize: whether to normalize the weights around 0 to be able to plot them as an image
	# n_filters: the number of filters from a layer we want to visualize
	# save_img:boolean indicating whether to save the output
	# path: str defining the path and name of hte output file


    # retrieve weights from the second hidden layer
    filters, biases = model.layers[layer].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    if normalize:
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    ix = 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(len(f[0,0,:])):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, len(f[0,0,:]), ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

