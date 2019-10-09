from __future__ import division

from os import walk
import os
import extra_functions
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt

batch_size =8

def read_model(cross=''):
    json_name = 'architecture_8_50_buildings_3_' + cross + '.json'
    weight_name = 'model_weights_8_50_buildings_3_' + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model

model = read_model()

data_path = 'Images_Without_Annotation'
num_channels = 3
num_mask_channels = 1
threshold = 0.3

ids = []
test_ids = []
for (dirpath, dirnames, filenames) in walk(data_path):
    ids.extend(filenames)
    break

for i in ids:
    if 'jpg' in i:
        test_ids.append(i)
print("Number of images: ",len(test_ids))

result = []

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


for image_id in test_ids:
    print("Predicting: ", image_id)
    image = extra_functions.read_image_test(image_id)
    predicted_mask = extra_functions.make_prediction_cropped(model, image, batch_size, size=(288, 288))
    
    image_v = flip_axis(image, 0)
    predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v,batch_size, size=(288,288))
    
    image_h = flip_axis(image, 1)
    predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h,batch_size, size=(288,288))
    
    image_s = image.swapaxes(0, 1)
    predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s,batch_size,  size=(288,288))
    new_mask = np.power(predicted_mask *
                        flip_axis(predicted_mask_v, 0) *
                        flip_axis(predicted_mask_h, 1) *
                        predicted_mask_s.swapaxes(0, 1), 0.25)
    new_mask[new_mask >= threshold] = 1;
    new_mask[new_mask < threshold] = 0;
    """code to save the predicted image as jpg"""
    plt.imsave("predicted_images/"+image_id, np.squeeze(new_mask,-1)*255,cmap='gray',dpi=1)
    
    
    alpha=0.6   
    new_mask = np.squeeze(new_mask,-1)
    color_mask = np.dstack((new_mask, new_mask, new_mask))
    
    image = image - image*(color_mask*0.3)
    image[:,:,0] += ((color_mask*255)*0.3)[:,:,0]
    
    plt.imsave("predicted_images/overlays/overlay_"+image_id,image,dpi=1)
