import streamlit as st
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from training.config_training import config
sys.path.append('../preprocessing/')
from preprocessing.step1 import *
from preprocessing.full_prep import lumTrans
from layers import nms,iou

sys.path.append('../training/')
sys.path.append('../../../AppData/Roaming/JetBrains/PyCharm2022.1/')

st.title('PyTorch Style Transfer')
img = np.load('./prep_result/0b8fdd1863b4362453001a94123ab40f.dcm_clean.npy')
# pbb = np.load('./bbox_result/d7850f462f15f4e8a21bf883450a505c_pbb.npy')

# pbb = pbb[pbb[:,0]>-1]
# pbb = nms(pbb,0.05)
# box = pbb[0].astype('int')[1:]
ax = plt.subplot(1,1,1)
plt.imshow(img[0],'gray')
plt.axis('off')
rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
ax.add_patch(rect)
