#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from segmentation_models import Unet
from keras.layers import Input, Conv2D
from keras.models import Model
from PIL import Image
import os
import keras
import argparse
from keras_radam import RAdam


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int)
parser.add_argument('--dropout',type=float)
parser.add_argument('--backbone')
parser.add_argument('--epochs',type=int)
parser.add_argument('--target')
parser.add_argument('--model')
parser.add_argument('--checkpoint')
parser.add_argument('--optimizer')
parser.add_argument('--lr',type=float)
parser.add_argument('--loss')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epochs
DROPOUT = args.dropout
BACKBONE= args.backbone

exp = args.target
x_train_dir = exp + '/img'
y_train_dir = exp + '/ann'

x_valid_dir = exp + '/img_sample'
y_valid_dir = exp + '/ann_sample'


CLASSES = ['foreground']


import segmentation_models as sm

model_args = dict(backbone_name=BACKBONE,
                  activation = 'relu',
                  encoder_weights=None,
                  input_shape=(None, None, 5))

if args.model == 'Linknet':
    model = sm.Linknet(**model_args)
elif args.model == 'Unet':
    model = sm.Unet(**model_args)
elif args.model == 'FPN':
    model = sm.FPN(**model_args,pyramid_dropout = DROPOUT, classes=1)
elif args.model == 'PSPNet':
    model = sm.PSPNet(**model_args,psp_dropout = DROPOUT, classes=1)

from rectified_adam import RectifiedAdam
from tf_rectified_adam import RectifiedAdam

if args.optimizer == 'Adam':
    optimizer = keras.optimizers.Adam(learning_rate=LR,amsgrad=True)
if args.optimizer == 'Nadam':
    optimizer = keras.optimizers.Nadam(LR)
elif args.optimizer == 'SGD':
    optimizer = keras.optimizers.SGD(LR)
elif args.optimizer == 'RAdam':
    optimizer = RAdam() # RAdam(learning_rate=0.001)

if args.loss == 'jdf':
    loss = sm.losses.JaccardLoss() + sm.losses.DiceLoss() + sm.losses.BinaryFocalLoss()
elif args.loss == 'jd':
    loss = sm.losses.JaccardLoss() + sm.losses.DiceLoss()
elif args.loss == 'df':
    loss = sm.losses.DiceLoss() + sm.losses.BinaryFocalLoss()
elif args.loss == 'jf':
    loss = sm.losses.JaccardLoss() + sm.losses.BinaryFocalLoss()

    
model.compile(optimizer,
              loss,
              [sm.metrics.IOUScore(threshold=0.5),
               sm.metrics.FScore(threshold=0.5)]
)
#sm.losses.BinaryCELoss() +  +
# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(args.checkpoint, save_weights_only=True, save_best_only=True, monitor='val_loss',mode='min'),
    #keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=3,min_delta=0.001),
]

# classes for data loading and preprocessing
class Dataset:
    CLASSES=['background','foreground']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        #self.class_values = [0,1]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = rasterio.open(self.images_fps[i])
        red = image.read(1)
        gren = image.read(2)
        blu = image.read(3)
        xx= image.read(4)
        yy= image.read(5)
        image = np.dstack((blu,gren,red,xx,yy))
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

#from keras import utils as Sequence
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# Dataset for train images
train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
    )

# Dataset for validation images
valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
    )

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# train model
import math
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=math.ceil(len(os.listdir(x_train_dir))/BATCH_SIZE),
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=math.ceil(len(os.listdir(x_valid_dir))/BATCH_SIZE),
)
