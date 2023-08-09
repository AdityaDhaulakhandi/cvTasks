import torch,torchvision
import eval
import numpy as np

def convert_mask_to_singleChannel(grayscale_mask,label_map):
    # Create a boolean mask of shape (num_classes, height, width)
    mask = (grayscale_mask[..., None] == label_map.reshape(1, 1, -1))

    # Use np.argmax to find the index of the first True value along the last axis
    target = np.argmax(mask, axis=-1)

    mask_tensor = torch.from_numpy(target) #convert to tensor
    return mask_tensor.unsqueeze(0)


def convert_mask_to_BinaryMasks(num_classes,grayscale_mask,label_map):
    shape = ( num_classes, grayscale_mask.shape[0], grayscale_mask.shape[1])
    target = np.zeros(shape,dtype='float32')

    for i in range(num_classes):
        target[i, :, :] = (grayscale_mask == label_map[i])

    # put 1 prob, for transparent channel (2) where we have a window channel(3) value 1
    # target[3,:,:] += target[2,:,:] 

    mask_tensor = torch.from_numpy(target)
    return mask_tensor


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


def load_trained_model(dataLoaders,result_dir,device,cfg):
    
    if cfg['backbone']['name'] == 'resnet50':
      from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights as pretrained_weights
      from torchvision.models.segmentation import deeplabv3_resnet50 as backbone_network
    else:
        from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as pretrained_weights
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as backbone_network
        
    weights = pretrained_weights.DEFAULT
    model = backbone_network(weights)
    model.classifier[-1] = torch.nn.Conv2d(256, cfg['num_classes'], kernel_size=1)
    
    model.load_state_dict(torch.load(cfg['backbone']['name']+'_'+cfg['GT_type']+'_state_dict'))
    
    # print(model)
    model.to(device)    
        
    mean_iou,iou_list = eval.get_results(result_dir,model,device,dataLoaders['test'])
    print("------Mean IoU------  : ",mean_iou)
    print()
    print(iou_list)
    print()
    print("----------------------------------Done!! Check results directory for results----------------------------------------")


  
def save_masks(idx,out,images,result_dir,device):
    
    # Add the window area to the tranparent
    # out[:,3,:,:] = out[:,3,:,:] + out[:,2,:,:] 

    # Channel-wise analysis
    # for i in range(out.shape[0]):
    #     for j in range(out.shape[1]):
    #         # print(j)
    #         torchvision.utils.save_image(out[:,j], f"{result_dir}/pred{ ((idx*out.shape[0])+i)}_Channel_{j}.jpeg")
    
    COLORMAP = [
      [1,1,1],    #background
      [128,0,0],    #plate 
      [0,128,0],    #window
      [128,128,0],  #transparent
      ]
    
    class_to_color = torch.zeros(size=(len(COLORMAP),3))
    for i,color in enumerate(COLORMAP):
        class_to_color[i] = torch.tensor(color) #colors for each class
    
    output_mask = torch.zeros(out.shape[0], 3, out.size(-2), out.size(-1), dtype=torch.float).to(device)
    for class_idx, color in enumerate(class_to_color):
        mask = out[:,class_idx,:,:] == torch.max(out, dim=1)[0].to(device)
        mask = mask.unsqueeze(1) 
        curr_color = color.reshape(1, 3, 1, 1).to(device)
        segment = mask*curr_color 
        output_mask += segment
    
    for i in range(images.shape[0]):
        torchvision.utils.save_image(output_mask[i], f"{result_dir+'/masks'}/pred_{ ((idx*images.shape[0])+i)}.jpeg")
        torchvision.utils.save_image((images[i].float()*(output_mask[i]))/255.0, f"{result_dir+'/images_with_mask'}/pred_{ ((idx*output_mask.shape[0])+i)}.jpeg")
        
        
def meanIOU(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return
        
    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return
    
    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        
        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else :
            iou_score = intersection / union
        iousum +=iou_score
        
    miou = iousum/target.shape[0]
    return miou