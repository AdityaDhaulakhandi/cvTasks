# Execution command --  CUDA_VISIBLE_DEVICES=1 python run.py 100 0 16 data/licenseplate_1200_200_200 1
#                       select the gpu = index               epochs resume batch_size data_dir load_model

import torch
import os
import dataLoader,train,utils

import sys
epochs = int(sys.argv[1])
resume =int(sys.argv[2])  # 1 to resume from a checkpoint
batch_size = int(sys.argv[3])
data_dir = sys.argv[4]
load_model =  int(sys.argv[5])


# test_dir = os.path.join(data_dir, 'test/Images/')

image_size = (320,460)
num_classes = 2  # Set the number of classes : plate and background


result_dir = 'results/'
if not os.path.isdir(result_dir):
      os.system('ln -s /shika_home/adityad22/prj1/results/ results')


# Loading the model
if( load_model == 1):
      print("---------LOADING THE TRAINED MODEL----------")
      utils.load_trained_model(data_dir,batch_size,result_dir,image_size)

else:
      print("-----------TRAINING THE MODEL-----------")
      print("Total Epochs : ",epochs,"Batch_size : ",batch_size,"Data directory : ",data_dir)

      # DEEPLAB BACKBONE
      from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as backbone_network
      from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as pretrained_weights
      weights = pretrained_weights.DEFAULT
      model = backbone_network(weights)
 
      loss_fn = torch.nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # lower the learning rate
      transform = weights.transforms(resize_size=None)
      # change the classifier at the end of the model
      model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
      dataLoaders = dataLoader.get_dataLoader(data_dir,batch_size,transform,image_size)

      torch.cuda.empty_cache()
      model= torch.nn.DataParallel(model)
      model.to(device)
      
      print('--------------------------------STARTING ---------------------------------------')
      model,best_loss = train.train(model,resume,epochs,optimizer,loss_fn,dataLoaders,device)  # 0 used for training from the start, enter the epoch where failed

      print("Loss ---- ",best_loss)

      torch.save(model,f='results/model')
      torch.save(model.module.state_dict(), 'model_stat_dict')

      print("------------- Saved the model !!! ---------------")

      # utils.load_trained_model(data_dir,batch_size,result_dir,image_size)