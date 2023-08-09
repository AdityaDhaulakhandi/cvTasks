import torch
import os
import dataloader,train,utils
import yaml


# User input -- hyperparamters
import argparse # Create the parser
parser = argparse.ArgumentParser()# Add an argument
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--epochs',type=int,help='Number of Epochs for training')
group.add_argument('--inference',type=int,help='1 for inference mode; 0 for training mode')
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--data_dir', type=str,help='Path to the data directory', required=True)
parser.add_argument('--resume', type=int, required=False)

args = parser.parse_args()
epochs = args.epochs
load_model = args.inference
batch_size = args.batch_size
data_dir = args.data_dir
resume = args.resume

# ---- Config file loading
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

#hyperparameters from config file
result_dir = cfg["result_dir"]
image_size = tuple(cfg["image_size"])
num_classes = cfg["num_classes"]

if not os.path.isdir(result_dir):
     os.system('ln -s /shika_home/adityad22/prj2/results/ results')

masks_dir = result_dir+'masks/'
if not os.path.isdir(masks_dir):
      os.mkdir(masks_dir)
images_with_mask_dir =  result_dir+'images_with_mask/'
if not os.path.isdir(images_with_mask_dir):
      os.mkdir(images_with_mask_dir)
    
#Load the backbone architecture pretrained weights
  
if cfg['backbone']['name'] == 'resnet50':
      from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights as pretrained_weights
else:
      from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as pretrained_weights
      
weights = pretrained_weights.DEFAULT
transform = weights.transforms(resize_size=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataLoaders = dataloader.get_dataLoader(data_dir,batch_size,transform,image_size,num_classes,cfg['GT_type'])      

# Loading the model
if( load_model == 1):      
      print("---------LOADING THE TRAINED MODEL ",cfg['backbone']['name']," ---------- ")
      utils.load_trained_model(dataLoaders,result_dir,device,cfg)

else:
      print("-----------TRAINING THE MODEL-----------")      
      print("Total Epochs : ",epochs,"Batch_size : ",batch_size,"Data directory : ",data_dir)

      # DEEPLAB BACKBONE
      if cfg['backbone']['name'] == 'resnet50': 
            from torchvision.models.segmentation import deeplabv3_resnet50 as backbone_network
      else:
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as backbone_network
            
      model = backbone_network(weights)
      
      # Load the weights from the license plate model
      if cfg['backbone']['load_trained_wts']:
            model.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=1)
            
            # load the parameters and load the model with them
            old_stat_dict = torch.load('state_dict_1',map_location=device)
            model.load_state_dict(old_stat_dict)
            
            # change the classifier to num_classes
            model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

            #---- Approach : copy the parameters to the plate class to plate classes 
            model.state_dict()['classifier.4.weight'][0] = old_stat_dict['classifier.4.bias'][0].detach().clone()
            model.state_dict()['classifier.4.bias'][0] = old_stat_dict['classifier.4.bias'][0].detach().clone()
                  
            for channel in range(1,num_classes):
                  model.state_dict()['classifier.4.weight'][channel] = old_stat_dict['classifier.4.bias'][1].detach().clone()
                  model.state_dict()['classifier.4.bias'][channel] = old_stat_dict['classifier.4.bias'][1].detach().clone()
       
      else : 
            model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
           
      optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # lower the learning rate
      
      torch.cuda.empty_cache()
      
      model = torch.nn.DataParallel(model)
      model.to(device)
      print('--------------------------------STARTING ---------------------------------------')
      model,best_loss = train.train_model(model,resume,epochs,optimizer,dataLoaders,device)
      
      print("Loss ---- ",best_loss)

      torch.save(model.module.state_dict(),cfg['backbone']['name']+'_'+cfg['GT_type']+'_state_dict')

      print("------------- Saved the model !!! ---------------")
