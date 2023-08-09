# to load the dataset
import torchvision
from torch.utils.data import Dataset, DataLoader


from PIL import Image
import os

class LicensePlateDataset(Dataset):
  def __init__(self,root_dir,image_size,phase,transformer=None):
    self.root_dir = os.path.join(root_dir, phase)
    print(self.root_dir)
    self.image_dir = os.path.join(self.root_dir, 'Images')
    self.mask_dir = os.path.join(self.root_dir, 'Masks')
    self.image_filenames = os.listdir(self.image_dir)
    self.transformer = transformer
    self.image_size = image_size

  def __len__(self):
    return len(self.image_filenames)


  def __getitem__(self, index):
    image_path = os.path.join(self.image_dir, self.image_filenames[index])
    mask_path = os.path.join(self.mask_dir, self.image_filenames[index])
    # print(mask_path,image_path)
    
    image = Image.open(image_path).convert('RGB')
    target = Image.open(mask_path)
    # print(type(image),type(target))

    transform_resize = torchvision.transforms.Resize(self.image_size)
    resized_image,resized_target = transform_resize(image), transform_resize(target)
    resized_target = torchvision.transforms.ToTensor()(resized_target)

    target_tensor = resized_target.long()
    
    # Apply any preprocessing
    if self.transformer is not None:
        resized_image = self.transformer(resized_image)
        
    return resized_image, target_tensor


def get_dataLoader(data_dir,batch_size,transform,image_size):  
      train_dataset = LicensePlateDataset(data_dir,image_size,phase='train',transformer=transform)
      val_dataset = LicensePlateDataset(data_dir,image_size,phase='val',transformer=transform)
      test_dataset = LicensePlateDataset(data_dir,image_size,phase='test',transformer=transform)
      
      train_loader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=2, pin_memory=True)
      val_loader = DataLoader(val_dataset, batch_size,num_workers=2, pin_memory=True)
      test_loader = DataLoader(test_dataset, batch_size=8,num_workers=2, pin_memory=True)

      dataLoader = {'train':train_loader, 'val':val_loader,'test':test_loader}

      return dataLoader
