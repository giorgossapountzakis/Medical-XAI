# -*- coding: utf-8 -*-
"""explainability.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bvl-iIF5dVSVx2OP8UPtK7NLT96RsvkJ

# imports
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import PIL.Image
import matplotlib.cm as cm
from IPython.display import Image, display
from collections import OrderedDict
import cv2
import itertools
import matplotlib.image

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import copy
from torchvision.models import *
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,RawScoresOutputTarget #an to kaleso me ari8mo eksigi gia tin katigoria ayti
from pytorch_grad_cam.utils.image import show_cam_on_image ,deprocess_image,preprocess_image
from pytorch_grad_cam.metrics.road import ROADCombined


# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

"""# dataset handling"""

labels=['COVID-19','Non-COVID','normal']

train_data_path='/data/data1/users/el17074/my_full_data/Lung Segmentation Data/Train'
valid_data_path='/data/data1/users/el17074/my_full_data/Lung Segmentation Data/Val'
test_data_path='/data/data1/users/el17074/my_full_data/Lung Segmentation Data/Test'
small_test_data_path='/data/data1/users/el17074/my_full_data/Lung Segmentation Data/Small_Test'

"""# normalization"""

norm_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.5, 0.5, 0.5],[0.224, 0.224, 0.224]),
                                       ])

"""# data loading"""

my_transforms = norm_transforms
image_datasets = {x: datasets.ImageFolder('/data/data1/users/el17074/my_data/Lung Segmentation Data/'+x, transform=my_transforms) for x in ['Train','Test', 'Val','Small_Test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['Train', 'Val']}
testdata_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['Test']}
Small_testdata_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=False) for x in ['Small_Test']}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""# model selection(change here for fcmiddlenumber and .fc)"""

path='/data/data1/users/el17074/mymodels/resnet/resnet50gridsearched/'
fcmiddlenumber=512#model0_params['fcmiddlenumber']#500
model1= models.resnet50()
num_ftrs = model1.fc.in_features #classifier[0] gia vgg, classifier gia densenet,fc giaresnet
model1.fc =  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))
model1.load_state_dict(torch.load(path+'weights.pt'))
model1=model1.to(device)
model1.eval()


'''


"""# predictions (change here depending on parameters)"""

#kanei gia 15 batch
count=0
allclasses=torch.Tensor()
allinputs=torch.Tensor()
for inputs, classes in testdata_dict['Test']:
  allclasses= torch.cat((allclasses, classes), 0)
  allinputs= torch.cat((allinputs, inputs), 0)
  count=count+1
  if count==15:
    break
model1 = model1.to(device)
allinputs=allinputs.to(device)
with torch.no_grad(): 
  outputs=model1(allinputs)
  _, preds = torch.max(outputs, 1)
  preds=preds.cpu().numpy()
  classes=allclasses.numpy()

del allinputs

confusion = confusion_matrix(classes, preds)
print('Confusion Matrix\n')
print(confusion)
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(classes, preds)))
print('Micro Precision: {:.2f}'.format(precision_score(classes, preds, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(classes, preds, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(classes, preds, average='micro')))
print('Macro Precision: {:.2f}'.format(precision_score(classes, preds, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(classes, preds, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(classes, preds, average='macro')))
print('Weighted Precision: {:.2f}'.format(precision_score(classes, preds, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(classes, preds, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(classes, preds, average='weighted')))
print('\nClassification Report\n')
classification_report=classification_report(classes, preds, target_names=['COVID-19','Non-COVID','Normal'])
print(classification_report)

f = open(path+'readme.txt', "w")

f.write(path)

f.write(np.array2string(confusion, separator=', '))
f.write('\nAccuracy: {:.2f}\n'.format(accuracy_score(classes, preds)))
f.write('Micro Precision: {:.2f}'.format(precision_score(classes, preds, average='micro')))
f.write('Micro Recall: {:.2f}'.format(recall_score(classes, preds, average='micro')))
f.write('Micro F1-score: {:.2f}\n'.format(f1_score(classes, preds, average='micro')))
f.write('Macro Precision: {:.2f}'.format(precision_score(classes, preds, average='macro')))
f.write('Macro Recall: {:.2f}'.format(recall_score(classes, preds, average='macro')))
f.write('Macro F1-score: {:.2f}\n'.format(f1_score(classes, preds, average='macro')))
f.write('Weighted Precision: {:.2f}'.format(precision_score(classes, preds, average='weighted')))
f.write('Weighted Recall: {:.2f}'.format(recall_score(classes, preds, average='weighted')))
f.write('Weighted F1-score: {:.2f}'.format(f1_score(classes, preds, average='weighted')))
f.write(classification_report)

f.close()



g = open(path+'times.txt', "w")
start_time_file = time.time()

'''

#1 batch prediction
inputs, classes = next(iter(Small_testdata_dict['Small_Test']))
rawinputs=inputs
model1 = model1.to(device)
inputs=inputs.to(device)
with torch.no_grad(): 
  outputs=model1(inputs)
  _, preds = torch.max(outputs, 1)
  preds=preds.cpu().numpy()
  classes=classes.numpy()


#g.write(str("Batch Prediction %s seconds ---" % (time.time() - start_time_file)))
#start_time_file = time.time()

print(preds)
print(classes)

foundcov=False
foundnoncov=False
foundnormal=False
covposition,noncovposition,normalposition=None,None,None
i=0
for item in classes:
  if foundcov==False or foundnoncov==False or foundnormal==False:
    if classes[i]==0 and foundcov==False:
      covposition=i
      foundcov=True
    elif classes[i]==1 and foundnoncov==False:
      noncovposition=i
      foundnoncov=True
    elif classes[i]==2 and foundnormal==False:
      normalposition=i
      foundnormal=True
    i=i+1
print(covposition,noncovposition,normalposition)



"""# Basic GradCAM"""

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,RawScoresOutputTarget #an to kaleso me ari8mo eksigi gia tin katigoria ayti
from pytorch_grad_cam.utils.image import show_cam_on_image ,deprocess_image,preprocess_image
from pytorch_grad_cam.metrics.road import ROADCombined

'''


print('Basic GradCAM')
start_time = time.time()
model = model1
target_layers = [model.layer4[-1]]#[model.layer4[-1]]#[model.layer4[-1]]
with GradCAM(model=model, target_layers=target_layers,use_cuda=True) as cam:
  im = transforms.ToPILImage()(inputs[covposition]).convert('RGB')
  im = np.asarray(im, dtype="float32" )/255
  input_tensor = inputs
  targets = None #xrisimopoiei ta preds oysiastika
  grayscale_cam= cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=True)
  visualization0 = show_cam_on_image(im, grayscale_cam[covposition], use_rgb=True)
  visualization0 = transforms.ToPILImage()(visualization0)
  visualizationcov=visualization0
#display(visualization0)




input_img = preprocess_image(im)
gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)#True
gb = gb_model(input_img, target_category=None)

cam_mask = cv2.merge([grayscale_cam[covposition], grayscale_cam[covposition], grayscale_cam[covposition]])
cam_gb = deprocess_image(cam_mask*gb)
gb = deprocess_image(gb)
gb = transforms.ToPILImage()(gb)
#display(gb)
cam_gb = transforms.ToPILImage()(cam_gb)
#display(cam_gb)
fig, ax = plt.subplots(1,4, figsize=(25, 25))
ax[0].imshow(rawinputs[covposition].permute(1,2,0))
ax[0].set_title('Original Image (COVID-19)')
ax[1].imshow(visualization0)
ax[1].set_title('GradCAM')
ax[2].imshow(gb)
ax[2].set_title('GuidedBackprop')
ax[3].imshow(cam_gb)
ax[3].set_title('GradCAM+GuidedBackprop')
plt.savefig(path+'gradcam+guidedbackprop_COVID-19.png')


with GradCAM(model=model, target_layers=target_layers,use_cuda=True) as cam:
  im = transforms.ToPILImage()(inputs[noncovposition]).convert('RGB')
  im = np.asarray(im, dtype="float32" )/255
  input_tensor = inputs
  targets = None #xrisimopoiei ta preds oysiastika
  grayscale_cam= cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=True)
  visualization0 = show_cam_on_image(im, grayscale_cam[noncovposition], use_rgb=True)
  visualization0 = transforms.ToPILImage()(visualization0)
#display(visualization0)

input_img = preprocess_image(im)
gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)#True
gb = gb_model(input_img, target_category=None)

cam_mask = cv2.merge([grayscale_cam[noncovposition], grayscale_cam[noncovposition], grayscale_cam[noncovposition]])
cam_gb = deprocess_image(cam_mask*gb)
gb = deprocess_image(gb)
gb = transforms.ToPILImage()(gb)
#display(gb)
cam_gb = transforms.ToPILImage()(cam_gb)
#display(cam_gb)
fig, ax = plt.subplots(1,4, figsize=(25, 25))
ax[0].imshow(rawinputs[noncovposition].permute(1,2,0))
ax[0].set_title('Original Image (Non-COVID)')
ax[1].imshow(visualization0)
ax[1].set_title('GradCAM')
ax[2].imshow(gb)
ax[2].set_title('GuidedBackprop')
ax[3].imshow(cam_gb)
ax[3].set_title('GradCAM+GuidedBackprop')
plt.savefig(path+'gradcam+guidedbackprop_Non-COVID.png')

with GradCAM(model=model, target_layers=target_layers,use_cuda=True) as cam:
  im = transforms.ToPILImage()(inputs[normalposition]).convert('RGB')
  im = np.asarray(im, dtype="float32" )/255
  input_tensor = inputs
  targets = None #xrisimopoiei ta preds oysiastika
  grayscale_cam= cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=True)
  visualization0 = show_cam_on_image(im, grayscale_cam[normalposition], use_rgb=True)
  visualization0 = transforms.ToPILImage()(visualization0)
#display(visualization0)

input_img = preprocess_image(im)
gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)#True
gb = gb_model(input_img, target_category=None)

cam_mask = cv2.merge([grayscale_cam[normalposition], grayscale_cam[normalposition], grayscale_cam[normalposition]])
cam_gb = deprocess_image(cam_mask*gb)
gb = deprocess_image(gb)
gb = transforms.ToPILImage()(gb)
#display(gb)
cam_gb = transforms.ToPILImage()(cam_gb)
#display(cam_gb)
fig, ax = plt.subplots(1,4, figsize=(25, 25))
ax[0].imshow(rawinputs[normalposition].permute(1,2,0))
ax[0].set_title('Original Image (Normal)')
ax[1].imshow(visualization0)
ax[1].set_title('GradCAM')
ax[2].imshow(gb)
ax[2].set_title('GuidedBackprop')
ax[3].imshow(cam_gb)
ax[3].set_title('GradCAM+GuidedBackprop')
plt.savefig(path+'gradcam+guidedbackprop_Normal.png')





#kathe eikona me th problepsi kai to gradcam poy thn aitiologei
fig, ax = plt.subplots(8,4, figsize=(15, 15))
x,y=0,-1
for i in range(0,32):
  im = transforms.ToPILImage()(inputs[i]).convert('RGB')
  im = np.asarray(im, dtype="float32" )/255
  visualization = show_cam_on_image(im, grayscale_cam[i], use_rgb=True)
  visualization = transforms.ToPILImage()(visualization)
  
  if i%4==0:
    x=0
    y+=1
  ax[y,x].imshow(visualization)
  ax[y,x].set_title('class '+ str(preds[i]))
  x+=1
plt.savefig(path+'gradcam_on_batch.png')
print("--- %s seconds ---" % (time.time() - start_time))


g.write(str("Batch GradCAM %s seconds ---" % (time.time() - start_time_file)))
start_time_file = time.time()



"""# GradCAM Layer Comparison"""

print('GradCAM Layer comparison')
start_time = time.time()
target_layers = [model.layer3[-1]]#[model.layer4[-1]] #na dokimaso kai me .conv3 sto ka8ena
with GradCAM(model=model, target_layers=target_layers,use_cuda=True) as cam:
  im = transforms.ToPILImage()(inputs[covposition]).convert('RGB')
  im = np.asarray(im, dtype="float32" )/255
  input_tensor = inputs
  targets = None #xrisimopoiei ta preds oysiastika
  grayscale_cam= cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=True)
  visualization1 = show_cam_on_image(im, grayscale_cam[covposition], use_rgb=True)
  visualization1 = transforms.ToPILImage()(visualization1)

target_layers = [model.layer2[-1]]#[model.layer4[-1]]
with GradCAM(model=model, target_layers=target_layers,use_cuda=True) as cam:
  im = transforms.ToPILImage()(inputs[covposition]).convert('RGB')
  im = np.asarray(im, dtype="float32" )/255
  input_tensor = inputs
  targets = None #xrisimopoiei ta preds oysiastika
  grayscale_cam= cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=True)
  visualization2 = show_cam_on_image(im, grayscale_cam[covposition], use_rgb=True)
  visualization2 = transforms.ToPILImage()(visualization2)

target_layers = [model.layer1[-1]]#[model.layer4[-1]]
with GradCAM(model=model, target_layers=target_layers,use_cuda=True) as cam:
  im = transforms.ToPILImage()(inputs[covposition]).convert('RGB')
  im = np.asarray(im, dtype="float32" )/255
  input_tensor = inputs
  targets = None #xrisimopoiei ta preds oysiastika
  grayscale_cam= cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=True)
  visualization3 = show_cam_on_image(im, grayscale_cam[covposition], use_rgb=True)
  visualization3 = transforms.ToPILImage()(visualization3)

fig, ax = plt.subplots(1,5, figsize=(25, 25))
ax[0].imshow(rawinputs[0].permute(1,2,0))
ax[0].set_title('Original Image (COVID-19)')
ax[4].imshow(visualizationcov)
ax[4].set_title('GradCAM on block4 last layer')
ax[3].imshow(visualization1)
ax[3].set_title('GradCAM on block3 last layer')
ax[2].imshow(visualization2)
ax[2].set_title('GradCAM on block2 last layer')
ax[1].imshow(visualization3)
ax[1].set_title('GradCAM on block1 last layer')
plt.savefig(path+'gradcam_layer_comparison.png')
print("--- %s seconds ---" % (time.time() - start_time))

g.write(str("GradCAM Layers %s seconds ---" % (time.time() - start_time_file)))
g.close()

'''

#h=open(path+'techincs_times.txt', "w")

#GradCAM technics

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenGradCAM, FullGrad,EigenCAM,LayerCAM
category=0#for covid images
# Showing the metrics on top of the CAM : 
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    visualization = cv2.putText(visualization, "ROAD", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA) 
    visualization = cv2.putText(visualization, f"{score:.5f}", (40, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)    
    return visualization
    
def benchmark0(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=0):
    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ]

    cam_metric = ROADCombined(percentiles=[25,50,75])#20, 40, 60, 80
    targets = None#[ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputTarget(category)]#[ClassifierOutputSoftmaxTarget(category)]
    
    visualizations = []
    percentiles = [25,50,75]#, 50, 90
    for name, cam_method in methods:
        start_time = time.time()
        with cam_method:
            print(cam_method)
            attributions = cam_method(input_tensor=input_tensor, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[covposition] 
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score = scores[0]
        print('reached here')
        visualization = show_cam_on_image(im, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualization=transforms.ToPILImage()(visualization)
        visualization = np.array(visualization)
        visualizations.append(visualization)
        #h.write("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))
    return transforms.ToPILImage()(np.hstack(visualizations))

	       
def benchmark1(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=0):
    methods = [("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("HiResCAM", HiResCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("XGradCAM", XGradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ]

    cam_metric = ROADCombined(percentiles=[25,50,75])#20, 40, 60, 80
    targets = None#[ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputTarget(category)]#[ClassifierOutputSoftmaxTarget(category)]
    
    visualizations = []
    percentiles = [25,50,75]#, 50, 90
    for name, cam_method in methods:
        start_time = time.time()
        with cam_method:
            print(cam_method)
            attributions = cam_method(input_tensor=input_tensor, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[covposition] 
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score =scores[0]
        print('reached here')
        visualization = show_cam_on_image(im, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualization=transforms.ToPILImage()(visualization)
        visualization = np.array(visualization)
        visualizations.append(visualization)
        #h.write("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))
    return transforms.ToPILImage()(np.hstack(visualizations))


def benchmark2(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=0):
    methods = [("FullGrad", FullGrad(model=model, target_layers=target_layers, use_cuda=True)),
	       ("EigenCAM", EigenCAM(model=model, target_layers=target_layers, use_cuda=True)),
	       ("LayerCAM", LayerCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ]

    cam_metric = ROADCombined(percentiles=[25,50,75])#20, 40, 60, 80
    targets = None#[ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputTarget(category)]#[ClassifierOutputSoftmaxTarget(category)]
    
    visualizations = []
    percentiles = [25,50,75]#, 50, 90
    for name, cam_method in methods:
        start_time = time.time()
        with cam_method:
            print(cam_method)
            attributions = cam_method(input_tensor=input_tensor, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[covposition] 
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score =scores[0]
        print('reached here')
        visualization = show_cam_on_image(im, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualization=transforms.ToPILImage()(visualization)
        visualization = np.array(visualization)
        visualizations.append(visualization)
        #h.write("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))
    return transforms.ToPILImage()(np.hstack(visualizations))

im = transforms.ToPILImage()(inputs[covposition]).convert('RGB')
im = np.asarray(im, dtype="float32" )/255
input_tensor = inputs
model = model1
target_layers = [model.layer4[-1]]

#x=benchmark0(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)
#x=x.save(path+'Technics0.png')
#x=benchmark1(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)
#x=x.save(path+'Technics1.png')
#x=benchmark2(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)
#x=x.save(path+'Technics2.png')


#h.close()


h=open(path+'techincs_times_smoothed.txt', "w")
print('smoothed')
#GradCAM technics

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenGradCAM, FullGrad,EigenCAM,LayerCAM
category=0#for covid images
# Showing the metrics on top of the CAM : 
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    visualization = cv2.putText(visualization, "ROAD", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA) 
    visualization = cv2.putText(visualization, f"{score:.5f}", (40, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)    
    return visualization
    
def benchmark0(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=0):
    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ]

    cam_metric = ROADCombined(percentiles=[25,50,75])#20, 40, 60, 80
    targets = None#[ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputTarget(category)]#[ClassifierOutputSoftmaxTarget(category)]
    
    visualizations = []
    percentiles = [25,50,75]#, 50, 90
    for name, cam_method in methods:
        start_time = time.time()
        with cam_method:
            print(cam_method)
            attributions = cam_method(input_tensor=input_tensor, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[covposition] 
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score = scores[0]
        print('reached here')
        visualization = show_cam_on_image(im, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualization=transforms.ToPILImage()(visualization)
        visualization = np.array(visualization)
        visualizations.append(visualization)
        h.write("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))
    return transforms.ToPILImage()(np.hstack(visualizations))

	       
def benchmark1(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=0):
    methods = [("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("HiResCAM", HiResCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ("XGradCAM", XGradCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ]

    cam_metric = ROADCombined(percentiles=[25,50,75])#20, 40, 60, 80
    targets = None#[ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputTarget(category)]#[ClassifierOutputSoftmaxTarget(category)]
    
    visualizations = []
    percentiles = [25,50,75]#, 50, 90
    for name, cam_method in methods:
        start_time = time.time()
        with cam_method:
            print(cam_method)
            attributions = cam_method(input_tensor=input_tensor, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[covposition] 
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score =scores[0]
        print('reached here')
        visualization = show_cam_on_image(im, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualization=transforms.ToPILImage()(visualization)
        visualization = np.array(visualization)
        visualizations.append(visualization)
        h.write("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))
    return transforms.ToPILImage()(np.hstack(visualizations))


def benchmark2(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=0):
    methods = [("FullGrad", FullGrad(model=model, target_layers=target_layers, use_cuda=True)),
	       ("EigenCAM", EigenCAM(model=model, target_layers=target_layers, use_cuda=True)),
	       ("LayerCAM", LayerCAM(model=model, target_layers=target_layers, use_cuda=True)),
               ]

    cam_metric = ROADCombined(percentiles=[25,50,75])#20, 40, 60, 80
    targets = None#[ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputTarget(category)]#[ClassifierOutputSoftmaxTarget(category)]
    
    visualizations = []
    percentiles = [25,50,75]#, 50, 90
    for name, cam_method in methods:
        start_time = time.time()
        with cam_method:
            print(cam_method)
            attributions = cam_method(input_tensor=input_tensor, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[covposition] 
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score =scores[0]
        print('reached here')
        visualization = show_cam_on_image(im, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualization=transforms.ToPILImage()(visualization)
        visualization = np.array(visualization)
        visualizations.append(visualization)
        h.write("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))
    return transforms.ToPILImage()(np.hstack(visualizations))

im = transforms.ToPILImage()(inputs[covposition]).convert('RGB')
im = np.asarray(im, dtype="float32" )/255
input_tensor = inputs
model = model1
target_layers = [model.layer4[-1]]

x=benchmark0(input_tensor, target_layers, eigen_smooth=True, aug_smooth=True)
x=x.save(path+'Technics0_smoothed.png')
x=benchmark1(input_tensor, target_layers, eigen_smooth=True, aug_smooth=True)
x=x.save(path+'Technics1_smoothed.png')
x=benchmark2(input_tensor, target_layers, eigen_smooth=True, aug_smooth=True)
x=x.save(path+'Technics2_smoothed.png')


h.close()



'''

"""# GradCAM metrics"""

def visualize_score(visualization, score, name):
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    return visualization

# Now lets see how to evaluate this explanation:
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

# For the metrics we want to measure the change in the confidence, after softmax, that's why
# we use ClassifierOutputSoftmaxTarget.
targets = [ClassifierOutputSoftmaxTarget(classes[covposition])]#analoga tin katigoria poy einai h eikona
cam_metric = CamMultImageConfidenceChange()
scores, visualizations = cam_metric(input_tensor, grayscale_cam, targets, model, return_visualization=True)
score = scores[0]
visualization = visualizations[covposition].cpu().numpy().transpose((1, 2, 0))
visualization = deprocess_image(visualization)
print(f"The confidence increase percent: {100*score}")
print("The visualization of the pertubated image for the metric:")
visualization=transforms.ToPILImage()(visualization)
visualization = np.array(visualization)
visualization=visualize_score(visualization,score,'confidence increase percent:')
visualization=transforms.ToPILImage()(visualization)
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
print("Drop in confidence", DropInConfidence()(input_tensor, grayscale_cam, targets, model))
print("Increase in confidence", IncreaseInConfidence()(input_tensor, grayscale_cam, targets, model))
plt.imshow(visualization)
plt.savefig(path+'CamMultImageConfidenceChange.png')
#afairo ta shmeia poy exoyn mikro gradcam syntelesth


from pytorch_grad_cam.sobel_cam import sobel_cam
img=inputs[covposition].permute(1,2,0).cpu().numpy()

sobel_cam_grayscale = sobel_cam(np.uint8(img * 255))
thresholded_cam = sobel_cam_grayscale < np.percentile(sobel_cam_grayscale, 75)
cam_metric = CamMultImageConfidenceChange()
scores, visualizations = cam_metric(input_tensor, thresholded_cam, targets, model, return_visualization=True)
score = scores[0]
visualization = visualizations[covposition].cpu().numpy().transpose((1,2, 0))
visualization = deprocess_image(visualization)
print(f"The confidence increase: {score}")
print("The visualization of the pertubated image for the metric:")
sobel_cam_rgb = cv2.merge([sobel_cam_grayscale, sobel_cam_grayscale, sobel_cam_grayscale])
x=np.hstack((sobel_cam_rgb, visualization))
visualization=transforms.ToPILImage()(x)
plt.imshow(visualization)
plt.savefig(path+'Sobel_Comparison.png')

from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
cam_metric = ROADMostRelevantFirst(percentile=50)
scores, visualizations = cam_metric(input_tensor, grayscale_cam, targets, model, return_visualization=True)
score = scores[0]
visualization = visualizations[covposition].cpu().numpy().transpose((1, 2, 0))
visualization = deprocess_image(visualization)
print(f"The confidence increase when removing 50% of the image: {score}")

visualization=transforms.ToPILImage()(visualization)
visualization = np.array(visualization)
visualization=visualize_score(visualization,score,'Conf incr remove 25%:')
visualization=transforms.ToPILImage()(visualization)

cam_metric = ROADMostRelevantFirst(percentile=90)
scores, visualizations = cam_metric(input_tensor, grayscale_cam, targets, model, return_visualization=True)
score = scores[0]
visualization_10 = visualizations[covposition].cpu().numpy().transpose((1, 2, 0))
visualization_10 = deprocess_image(visualization_10)
print(f"The confidence increase when removing 10% of the image: {score}")
print("The visualizations:")

visualization_10=transforms.ToPILImage()(visualization_10)
visualization_10 = np.array(visualization_10)
visualization_10=visualize_score(visualization_10,score,'Conf incr remove 10%:')
visualization_10=transforms.ToPILImage()(visualization_10)

visualization = np.array(visualization)
visualization_10 = np.array(visualization_10)
visualization=np.hstack((visualization, visualization_10))
visualization=transforms.ToPILImage()(visualization)
plt.imshow(visualization)
plt.savefig(path+'ROAD.png')

from pytorch_grad_cam.metrics.road import ROADCombined
cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
scores = cam_metric(input_tensor, grayscale_cam, targets, model)
print(f"CAM, Combined metric avg confidence increase with ROAD accross 4 thresholds (positive is better): {scores[0]}")
f=open(path+'readme.txt','a')
f.write(f"CAM, Combined metric avg confidence increase with ROAD accross 4 thresholds (positive is better): {scores[0]}")
f.close





#LIME
print('LIME')
from lime import lime_image

def get_preprocess_transform():   
    transf = transforms.Compose([
        transforms.ToTensor()
    ])    

    return transf    
preprocess_transform = get_preprocess_transform()


def batch_predict(images):
    model2=model1
    model2.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model2 = model2.type(torch.cuda.FloatTensor).to(device)
    batch = batch.type(torch.cuda.FloatTensor).to(device)
    


    logits = model2(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

test_pred = batch_predict([np.array(inputs[0].permute(1,2,0).cpu())])
test_pred.squeeze().argmax()

image=np.array(inputs[0].permute(1,2,0).cpu()).astype('double')
#image=image.to(device)
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image,
                                         batch_predict,
                                         top_labels=3, 
                                         hide_color=None, #None
                                         num_samples=1000) # number of images that will be sent to classification function

from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp, mask)
plt.imshow(img_boundry1)
matplotlib.image.imsave(path+'lime0.png', img_boundry1)

fig, ax = plt.subplots(1,3, figsize=(25, 25))
for i in range(0,3):
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[i], positive_only=True, num_features=5, hide_rest=False)
  img_boundry = mark_boundaries(temp, mask)
  ax[i].imshow(img_boundry)
  x=i
  ax[i].set_title('Areas that contribute to prediction for class ' +str(i))
  #edo moy deixnei ta top positive
  plt.savefig(path+'lime1.png')

fig, ax = plt.subplots(1,3, figsize=(25, 25))
for i in range(0,3):
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[i], positive_only=False, num_features=5, hide_rest=False)
  img_boundry = mark_boundaries(temp, mask)
  ax[i].imshow(img_boundry)
  x=i
  ax[i].set_title('Areas that contribute to prediction for class ' +str(i))
#edo moy deixnei ta top genika, na paratiriso diafores
plt.savefig(path+'lime2.png')

fig, ax = plt.subplots(1,3, figsize=(25, 25))
for i in range(0,3):
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[i], positive_only=False, num_features=10, hide_rest=True)
  img_boundry = mark_boundaries(temp, mask)
  ax[i].imshow(img_boundry)
  x=i
  ax[i].set_title('Areas that contribute to prediction for class ' +str(i))
#edo moy deixnei ta top genika
plt.savefig(path+'lime3.png')




#SHAP
print('SHAP')
import shap

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


transform= [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

inv_transform= [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)

def predict(img: np.ndarray) -> torch.Tensor:
    model2=model1
    model2.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = nhwc_to_nchw(torch.Tensor(img))
    img = img.to(device)
    model2=model2.to(device)
    with torch.no_grad(): 
      output = model2(img)
    return output
# Check that transformations work correctly
class_names=['COVID-19','Non-COVID','Normal']
Xtr = transform(inputs)
out = predict(Xtr[0:10])
newclasses = torch.argmax(out, axis=1).cpu().numpy()
print(f'Classes: {newclasses}: {np.array(class_names)[newclasses]}')

topk = 3
batch_size = 32
n_evals = 1000


masker_blur = shap.maskers.Image("blur(128,128)", Xtr[0].shape)
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)


shap_values = explainer(Xtr[0:10], max_evals=n_evals, batch_size=batch_size,
                        outputs=shap.Explanation.argsort.flip[:topk])

shap_values.data = inv_transform(shap_values.data).cpu().numpy()
shap_values.values = [val for val in np.moveaxis(shap_values.values,-1, 0)]

shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names,
                width=30,aspect=0.4, hspace=0.3,
                )
plt.savefig(path+'shapley0.png')

'''


