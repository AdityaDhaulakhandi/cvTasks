# Perfomance metric for test  --- mAP and IOU
import torch
import torchvision
from torchvision.io.image import read_image
import utils
import os

def get_results(model,transform,device,test_folder,result_dir,image_size):

      model.eval()
     
      # Iterate over the test images
      for i,image_file in enumerate(os.listdir(test_folder)):
            # Load and preprocess the image
            image_path = os.path.join(test_folder, image_file)
            image_tensor = read_image( image_path )
            resized_image = torchvision.transforms.Resize(image_size, antialias=True)(image_tensor)

            # Apply the transform
            resized_image_tensor = transform(resized_image)
            resized_image_tensor = resized_image_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                  outputs = model(resized_image_tensor)['out']
                  normalized_output = torch.softmax(outputs,dim=1)

                  # print("normalized_output : ",normalized_output.shape,normalized_output.max().item(),normalized_output.min().item())

                  # print(normalized_output.get_device(), torch.arange(normalized_output.shape[1])[:, None, None, None].get_device())
                  all_classes_masks = normalized_output.argmax(dim=1) == torch.arange(normalized_output.shape[1])[:, None, None, None].to(device)
                  all_classes_masks = all_classes_masks.swapaxes(0, 1)
                  # print("all_classes_masks : ",all_classes_masks.shape)
                  # predicted_masks = torch.argmax(normalized_output,dim=1)

            utils.save_results(i,resized_image,all_classes_masks[0],result_dir)
            

def compute_IOU(model, test_loader,device,num_classes):
    from torchmetrics import JaccardIndex
    import numpy as np

    model.eval()
    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes)
    jaccard = jaccard.to(device)

    IoU = []
    with torch.no_grad():
      for images, masks in test_loader:
          images = images.to(device)
          targets = masks.to(device)
    
          outputs = model(images)['out']
          normalized_output = torch.softmax(outputs,dim=1)
          predictions = torch.argmax( normalized_output , dim=1)

      #     torch.argmax -- gives me a tensor of size batch_size,w,h giving me the indices of the largest per class 
      #     print(predictions.shape)

          iou = jaccard(predictions, targets.squeeze(1))
          iou = iou.detach().cpu().numpy()
          IoU.append(iou.item())
          # print(iou)

    mean_iou = np.mean(IoU)
    return mean_iou,IoU



           
def get_result(num_classes,result_dir,model,device,test_loader):
        from torchmetrics import JaccardIndex
        import numpy as np
        
        model.eval()
        jaccard = JaccardIndex(task="multiclass", num_classes=num_classes)
        jaccard = jaccard.to(device)

        IoU = []

        with torch.no_grad():
                for i, (images, masks) in enumerate(test_loader):
                        images = images.to(device)
                        targets = masks.to(device)
                        outputs = model(images)['out']
                        preds = torch.sigmoid(model(images)['out'])
                        out = (preds > 0.5).float()     
                        utils.save_masks(i,out,images,result_dir,device)
                        
                        normalized_output = torch.softmax(outputs,dim=1) 
                        final_predictions = normalized_output.argmax(dim=1)
                        iou = jaccard(final_predictions, targets.squeeze(1))
                        iou = iou.detach().cpu().numpy()
                        IoU.append(iou.item())
                        
        
        return np.mean(IoU),IoU
 