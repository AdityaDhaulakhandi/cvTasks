# Perfomance metric for test  --- mAP and IOU
import torch
import utils

              
def get_results(result_dir,model,device,test_loader):
        import numpy as np
        
        from torchmetrics import JaccardIndex
        jaccard = JaccardIndex(task="multiclass", num_classes=4)
        jaccard = jaccard.to(device)
        
        model.eval()
        IoU = []
        with torch.no_grad():
                for i, (true_images,images, masks,_) in enumerate(test_loader):
                        true_images = true_images.to(device)
                        images = images.to(device)
                        targets = masks.to(device)
                        # print(class_present_map)
                        # outputs = model(images)['out']
                        # out = (outputs > 0.5).float() 
                        preds = model(images)['out']
                        out = torch.softmax(preds,dim=1)
                        utils.save_masks(i,out,true_images,result_dir,device)
                        
                        if targets.shape[1] == 1:
                                final_predictions = out.argmax(dim=1)
                                
                                iou = jaccard(targets.squeeze(1), final_predictions)
                                iou = iou.detach().cpu().numpy()
                                # print(iou)
                                IoU.append(iou.item())
                        else:
                                iou = utils.meanIOU(targets,out)
                                IoU.append(iou)
                       
        return np.mean(IoU),IoU
                         
