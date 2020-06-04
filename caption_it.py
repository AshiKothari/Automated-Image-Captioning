#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing dependencies
import pandas as pd
import numpy as np
import keras
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

model = load_model('./model_weights/model_9.h5')
model._make_predict_function()

# In[36]:


modell = ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[37]:

modeln = Model(modell.input,modell.layers[-2].output)
modeln._make_predict_function()

# In[72]:

with open("./w2i.pkl","rb") as f:
    word_to_idx=pickle.load(f)
with open("./i2w.pkl","rb") as g:
    idx_to_word=pickle.load(g)


# In[73]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img


# In[74]:

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = modeln.predict(img)
    feature_vector = feature_vector.reshape((1,feature_vector.shape[1]))
    #print(feature_vector.shape)
    return feature_vector


# In[75]:


max_len=35
def predict_caption(photo):  
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)        
        if word == "endseq":
            break   
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[76]:

def captionis(img):
    enc=encode_image(img)
    cap=predict_caption(enc)
    return cap



# In[77]:


# In[ ]:




