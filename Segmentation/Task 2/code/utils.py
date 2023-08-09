import torch,torchvision
import eval

from torchvision.utils import save_image

import matplotlib.pyplot as plt
from PIL import Image

def checkpoint(model,epoch,optimizer,best_loss,filename):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            }, filename)

    
def resume_checkpoint(model,optimizer,filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'],checkpoint['loss']


def load_trained_model(data_dir,batch_size,result_dir,image_size):
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as pretrained_weights
    import dataLoader

    weights = pretrained_weights.DEFAULT
    num_classes = 2  # Set the number of classes : plate and background
    # data_dir = '/content/licenseplate_1200_200_200' #apth to the dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = weights.transforms(resize_size=None)

    model = torch.load('results/model')
    model.to(device)
    
    dataLoaders = dataLoader.get_dataLoader(data_dir,batch_size,transform,image_size)
    print('-----------------------------------------------------------------------------------------')
    mean_iou,iou_list = eval.get_result(num_classes,result_dir,model,device,dataLoaders['test'])
    print("------Mean IoU------  : ",mean_iou)
    print()
    print(iou_list)
    print()

    

    # mean_iou,iou_list = eval.compute_IOU(model,dataLoaders['test'],device,num_classes)
    # print("------Mean IoU------  : ",mean_iou)
    # print()
    # print(iou_list)
    # print()

    # eval.get_results(model,transform,device,test_dir,result_dir,image_size)
    # print("----------------------------------Done!! Check results directory for results----------------------------------------")


def save_results(i,image,mask,result_dir):
    image_with_masks = torchvision.utils.draw_segmentation_masks(image, mask)
    img = torchvision.transforms.ToPILImage()(image_with_masks)
    img.save(result_dir+'pred_'+str(i)+'.jpeg')
    
    
    
def save_masks(idx,out,images,result_dir,device):
    class_to_color = [torch.tensor([1, 1, 1]), torch.tensor([0,255,0])] #colors
    # print(images.shape)
    output_mask = torch.zeros(out.shape[0], 3, out.size(-2), out.size(-1), dtype=torch.float).to(device)
    for class_idx, color in enumerate(class_to_color):
        mask = out[:,class_idx,:,:] == torch.max(out, dim=1)[0].to(device)
        mask = mask.unsqueeze(1) # should have shape 1, 1, 100, 100
        curr_color = color.reshape(1, 3, 1, 1).to(device)
        segment = mask*curr_color # should have shape 1, 3, 100, 100
        output_mask += segment
  
    for i in range(output_mask.shape[0]):
        torchvision.utils.save_image(images[i,:,:,:]*output_mask[i,:,:,:], f"{result_dir}/pred_{ ((idx*output_mask.shape[0])+i)}.jpeg")
        
    