import torch,torchvision

from torch.utils.tensorboard import SummaryWriter
import utils
import numpy as np


def train_model(model,resume,epochs,optimizer,dataLoader,device):
  import copy
  writer = SummaryWriter()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 100.0
  start_epoch = 0

  if resume == 1:
    start_epoch, best_loss = utils.resume_checkpoint(model,optimizer, f"model_checkpoint.pth")

  for epoch in range(start_epoch, epochs):
      val_loss = 0.0
      train_loss = 0.0

      print('Epoch {}/{}'.format(epoch, epochs - 1))
      
      # Training phase 
      model.train()
      for _,images,targets,class_present_map in dataLoader['train']:
          
          images = images.float().to(device)
          targets = targets.to(device)
          # the class distribution for the training set
          class_dist = class_present_map[0].to(device)
          # class_present_map = class_present_map.to(device)
          optimizer.zero_grad()
          pred = model(images)['out']
          
          # Calculate the weights
          sorted_frequencies = sorted(class_dist, reverse=True)
          max_freq = sorted_frequencies[0]
          weights = [max_freq / freq for freq in sorted_frequencies]
          # Create the weight tensor
          weights_tensor = torch.tensor(weights).to(device)
          
          if targets.shape[1] == 1:
            loss = torch.nn.CrossEntropyLoss(weight=weights_tensor)(pred,targets.squeeze(1))
          else:
             loss = torch.nn.CrossEntropyLoss(weight=weights_tensor)(pred,targets)
          
          loss.backward()
          optimizer.step()
          
          train_loss += loss.item()*images.size(0)
          
      train_loss /= len(dataLoader['train'].dataset)
      writer.add_scalar("Loss/train", train_loss, epoch)
      print('Training Loss: {:.4f}'.format(train_loss))
      
      # Validation every 20 epochs
      if (epoch+1) % 40 == 0:
        print("-----------------------------")
        model.eval()
        
        with torch.no_grad():
          for _,images, targets,_ in dataLoader['val']:
            images = images.float().to(device)
            targets = targets.to(device)
            
            pred = model(images)['out']  
            normalized_pred = (torch.softmax(pred,dim=1)).float() 
            
            if targets.shape[1] == 1:
              loss = torch.nn.CrossEntropyLoss()(normalized_pred,targets.squeeze(1))
            else:
             loss = torch.nn.CrossEntropyLoss()(normalized_pred,targets)
          
            val_loss += loss.item()*images.size(0)

        val_loss /= len( dataLoader['val'].dataset)
        print('Validation Loss: {:.4f}'.format(val_loss))

        #--------------------- Checkpoint---------------------
        utils.checkpoint(model,epoch,optimizer,best_loss,f"model_checkpoint.pth")
        # #--------------------- Checkpoint---------------------
        
        if best_loss*10000 > val_loss*10000:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
      print('-' * 10)

  model.load_state_dict(best_model_wts)

  writer.flush()
  writer.close()
  return model,best_loss




