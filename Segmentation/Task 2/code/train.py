import torch 
from torch.utils.tensorboard import SummaryWriter
import utils

def train(model,resume,epochs,optimizer,loss_fn,dataLoader,device):
  import copy
  writer = SummaryWriter()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10.0
  start_epoch = 0

  if resume == 1:
    start_epoch, best_loss = utils.resume_checkpoint(model,optimizer, f"results/model_checkpoint.pth")

  for epoch in range(start_epoch, epochs):
      val_loss = 0.0
      train_loss = 0.0

      print('Epoch {}/{}'.format(epoch, epochs - 1))
      
      # Training phase 
      model.train()
      for images, targets in dataLoader['train']:
          images = images.to(device)
          targets = targets.to(device)
          optimizer.zero_grad()
          
          pred = model(images)['out']
          # print(pred.shape)
          # print( torch.max(pred).item(),torch.min(pred).item())
          # print("Softmax --",torch.max(torch.softmax(pred,dim=1)).item(),torch.min(torch.softmax(pred,dim=1)).item())
          
          loss = loss_fn( pred,targets.squeeze(1) ) # get a single channel from the mask

          loss.backward()
          optimizer.step()
          train_loss += loss.item() * images.size(0) # multiplied by batch_size

      train_loss /= len(dataLoader['train'].dataset)
      writer.add_scalar("Loss/train", train_loss, epoch)
      print('Training Loss: {:.4f}'.format(train_loss))
      
      # Validation every 8 epochs
      if (epoch+1) % 8 == 0:
        print("-----------------------------")
        model.eval()
        
        with torch.no_grad():
          for images, targets in dataLoader['val']:
            images = images.to(device)
            targets = targets.to(device)
            pred = model(images)['out']
            normalized_pred = torch.softmax(pred,dim=1)
            loss = loss_fn(normalized_pred, targets.squeeze(1)) # get a single channel from the mask

            val_loss += loss.item() * images.size(0) # multiplied by batch_size

        val_loss /= len( dataLoader['val'].dataset)
          
        print('Validation Loss: {:.4f}'.format(val_loss))

        #--------------------- Checkpoint---------------------
        utils.checkpoint(model,epoch,optimizer,best_loss,f"results/model_checkpoint.pth")
        #--------------------- Checkpoint---------------------
        
        if best_loss*10000 > val_loss*10000:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
      print('-' * 10)

  model.load_state_dict(best_model_wts)

  writer.flush()
  writer.close()
  return model,best_loss




