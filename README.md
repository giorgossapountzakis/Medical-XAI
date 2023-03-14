# Medical-XAI
Medical image classification networks and explainability technics comparison

In our study, we examine different such networks and evaluate their results in classifying chest X-rays as those of a coronavirus patient or a healthy individual. At the same time, we attempt to understand how the network arrived at this particular decision, through different explainability methods. The goal of the thesis is therefore the comparison between the different networks and methods of explainability.

Model training, grid search and masked training is performed in files 'model_name'.py  
Arithmetic metrics, GradCAM, GradCAM Layers, GradCAM methods is performed in files explainability_'model_name'.py  
GradCAM metrics, LIME, SHAP partition, gradient, kernel explainers is performed in files 'model_name'_extras.py  
Sobel comparison is performed in sobel.py  
The making of our masks is performed in maskmaker.py  
Arithmetic metrics, GradCAM, GradCAM Layers for masked models can be performed also in maskedtest.py  

Generally if the 'Model selection' part is changed in every file the code works for any different model. Also GradCAM target_layers should be changed from model.features[-1] in some cases and size from 224 to 299 in inception networks. Here are the changes altogether:     



  
    
```python     
path='~/efficientnethyperparameters2/'  
fcmiddlenumber=512  
model1= models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)  
num_ftrs = model1.classifier[1].in_features  
model1.classifier =  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  


path='~/efficientnethyperparameters/'  
fcmiddlenumber=512  		
model1= models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)  
num_ftrs = model1.classifier[1].in_features  
model1.classifier =  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  


path='~/squeezenetmasked/'  
fcmiddlenumber=0  		
model1= models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)  
model1.classifier[1] =  nn.Conv2d(512, 3, kernel_size=(1,1), stride=(1,1))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  


path='~/mobilenetv2hyperparameters/'  
fcmiddlenumber=512		#maybe change here  
model1= models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)  
num_ftrs = model1.classifier[1].in_features  
model1.classifier =  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  

path='~/densenet121gridsearched/'  
fcmiddlenumber=256#model0_params['fcmiddlenumber']  
model1= models.densenet121()  
num_ftrs = model1.classifier.in_features  
model1.classifier =  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  



#Change GradCAM target_layers= [model.layer4[-1]]  
path='~/resnet50hyperparameters2/'  
fcmiddlenumber=512#model0_params['fcmiddlenumber']  
model1= models.resnet50()  
num_ftrs = model1.fc.in_features  
model1.fc =  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  

#Change size from 224 to 299 everywhere and in GradCAM target_layers= [model.Mixed_7c.branch_pool]  
path='~/inceptionv3hyperparameters/'  
fcmiddlenumber=256   	
model1= models.inception_v3(weights=Inception_V3_Weights.DEFAULT)  
aux_num_ftrs = model1.AuxLogits.fc.in_features  
model1.AuxLogits.fc = nn.Linear(aux_num_ftrs, 3)  
num_ftrs = model1.fc.in_features  
model1.fc =  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  


path='~/vgg19masked/'  
fcmiddlenumber=256#model0_params['fcmiddlenumber']  
model1= models.vgg19()  
num_ftrs = model1.classifier[0].in_features  
model1.classifier=  nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, fcmiddlenumber)),('relu', nn.ReLU()),('dropout',nn.Dropout()),('fc2', nn.Linear(fcmiddlenumber, 3))]))  
model1.load_state_dict(torch.load(path+'weights.pt'))  
model1=model1.to(device)  
model1.eval()  
