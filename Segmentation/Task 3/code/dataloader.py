import torchvision,torch
import utils

from io import BytesIO
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import lmdb
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class lmdbDataset(Dataset):
    
    def __init__(self,
                 input_path,
                 clean_path,
                 image_size,
                 num_classes,
                 gt_type,
                 transform=None,
                 config=None,
                 phase=None,
                 name='lmdbDataset'):

        self.input_dir = input_path
        self.clean_dir = clean_path
        self.transform = transform
        self.config = config
        self.phase = phase #train,test,val
        self.image_size=image_size
        self.num_classes =num_classes
        self.gt_type = gt_type
        self.name = name
        
        # class weight
        self.class_frequencies = torch.tensor([0.,0.,0.,0.])
        
        # input image paths 
        self.input_images = glob.glob(os.path.join(self.input_dir, "*.*"), recursive=True)

        # get image names 
        self.image_names = [img_path.split(os.sep)[-1] for img_path in self.input_images
            if os.path.splitext(img_path)[-1].lower() in [".png", ".jpeg", ".jpg", ".bmp"]
        ]

        # create lmdb
        if self.config.lmdbdir:
            os.makedirs(self.config.lmdbdir, exist_ok=True)

            # name lmdb 
            self.lmdb_name = os.path.join(self.config.lmdbdir, self.phase + '_' + self.name + '.lmdb')
            if not os.path.exists(self.lmdb_name):
                # create lmdb 
                self.write_lmdb()

            self.db = lmdb.open(self.lmdb_name,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)

    def write_lmdb(self):
        '''
        Write the LMDB file corresponding to this dataset
        '''
        if not self.config.lmdbdir:
            raise Exception("LMDB dir is not provided!")

        db = lmdb.open(self.lmdb_name,
                       map_size=int(500 * 1e9),
                       readonly=False,
                       meminit=False,
                       map_async=True)

        write_frequency = 100

        print('***** Starting to write LMDB ' + self.lmdb_name + ' *****')
        print(f"Num samples: {len(self.image_names)}")

        max_idx = len(self.image_names)

        txn = db.begin(write=True)
        all_imagefiles = []

        # paired dataset 
        for idx in range(max_idx):
            imagefile = os.path.join(self.input_dir, self.image_names[idx])
            cleanfile = os.path.join(self.clean_dir, self.image_names[idx]) 
            all_imagefiles = [imagefile, cleanfile]
            
            for imgfile in all_imagefiles:
                fl_bytes = self._resize_encode(imgfile)
                key_str = u'{}'.format(imgfile).encode('ascii')
                if not txn.get(key_str):
                    txn.put(key_str, fl_bytes)

            # save image and clean files
            if idx > 0 and idx % write_frequency == 0:
                print("Writing LMDB txn to disk [%d/%d]" % (idx, max_idx))
                txn.commit()
                txn = db.begin(write=True)

        # finish remaining transactions
        print("Writing final LMDB txn to disk [%d/%d]" % (max_idx, max_idx))
        txn.commit()

        print("Flushing LMDB database")
        db.sync()
        db.close()
        print("**** LMDB writing complete")

    def _buffToImg(self, buff):
        '''
        encodes buffer to RGB image 
        '''
        nparr = np.frombuffer(buff, np.uint8)
        img = Image.open(BytesIO(nparr)).convert('RGB')
        img = np.array(img) #convert to numpy array for cv2 functionality
        return img

    def _resize_encode(self, imgfile):
        '''
        encode the RGB image to buffer
        '''
        with open(imgfile, 'rb') as fl:
            fl_bytes = fl.read()
        return fl_bytes

    def __len__(self):
        return len(self.image_names)
    
    # function to get the distribution of the class in the training dataset
    def get_class_distribution(self):
        total_masks = self.__len__()
        for index in range(total_masks):
            clean_image_name = os.path.join(self.clean_dir, self.image_names[index]) 
            if self.config.lmdbdir:
                # read from lmdb 
                with self.db.begin(write=False) as txn:
                    clean_img = self._buffToImg(txn.get(clean_image_name.encode('ascii')))
            else:
                # read from disk 
                clean_img = cv2.imread(clean_image_name, cv2.COLOR_BGR2RGB)

            # Convert the image to a single channel grayscale image
            grayscale_mask = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
            
            #pixel code in grayscale image for background,plate,window,tranparent
            label_map = np.array([0, 38, 75, 113])

            # used to tackle class imbalance problem
            class_present_map = torch.tensor([0.,0.,0.,0.])

            for cls in range(self.num_classes):
                if (label_map[cls]==np.unique(grayscale_mask)).any():
                        class_present_map[cls] = 1.
            
            self.class_frequencies += (class_present_map)  
            
    def __getitem__(self, index):
        
        input_image_name = os.path.join(self.input_dir, self.image_names[index])
        clean_image_name = os.path.join(self.clean_dir, self.image_names[index]) 
        
        if self.config.lmdbdir:
            # read from lmdb 
            with self.db.begin(write=False) as txn:
                input_img = self._buffToImg(txn.get(input_image_name.encode('ascii')))
                clean_img = self._buffToImg(txn.get(clean_image_name.encode('ascii')))
        else:
            # read from disk 
            input_img = cv2.imread(input_image_name, cv2.COLOR_BGR2RGB)
            clean_img = cv2.imread(clean_image_name, cv2.COLOR_BGR2RGB)

    
        image_tensor = torch.from_numpy(input_img).permute(2,0,1)
    
        # Convert the image to a single channel grayscale image
        grayscale_mask = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
        
        #pixel code in grayscale image for background,plate,window,tranparent
        label_map = np.array([0, 38, 75, 113])


        #--------------------- Change for using different ground truth masks
        print(self.gt_type)
        if self.gt_type=='singleMask':
            mask_tensor = utils.convert_mask_to_singleChannel(grayscale_mask,label_map) 
        else:
            mask_tensor = utils.convert_mask_to_BinaryMasks(self.num_classes,grayscale_mask,label_map)
        
        resize_transform = torchvision.transforms.Resize(self.image_size,antialias=True)
        # print(image_tensor.shape,mask_tensor.shape)
        resized_image,resized_target = resize_transform(image_tensor),resize_transform(mask_tensor)

        # Apply any preprocessing
        if self.transform is not None:
            tranformed_image = self.transform(resized_image)
        
        return resized_image,tranformed_image,resized_target,self.class_frequencies


def get_train_dataloader(config,transform,image_size,num_classes,gt_type):
    
    dataset = lmdbDataset(input_path=config.input_train_path,
                         clean_path=config.clean_train_path,
                         image_size=image_size,
                         num_classes=num_classes,
                         gt_type=gt_type,
                         transform=transform,
                         config=config,
                         phase='train',
                         name=config.train_db_name)

    # get the class distribution
    dataset.get_class_distribution()
    
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.workers,
                            collate_fn=None,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=False)
    return dataloader

def get_val_dataloader(config,transform,image_size,num_classes,gt_type):
    
    dataset = lmdbDataset(input_path=config.input_val_path,
                         clean_path=config.clean_val_path,
                         image_size=image_size,
                         num_classes=num_classes,
                         gt_type=gt_type,
                         transform=transform,
                         config=config,
                         phase='val', 
                         name=config.val_db_name
    )

    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.workers,
                            collate_fn=None,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=False)
    return dataloader


def get_test_dataloader(config,transform ,image_size,num_classes,gt_type):
    
    dataset = lmdbDataset(input_path=config.input_test_path,
                         clean_path=config.clean_test_path,
                         image_size=image_size,
                         num_classes=num_classes,
                         gt_type=gt_type,
                         transform=transform,
                         config=config,
                         phase='test',
                         name=config.test_db_name)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=1,
                            collate_fn=None,
                            drop_last=False)

    return dataloader


def get_dataLoader(data_dir,batch_size,transform,image_size,num_classes,gt_type):
    from easydict import EasyDict
    # Base default config
    config = EasyDict({})
    print(data_dir)
        
    # trainset
    config.input_train_path = data_dir+'train/Images'
    config.clean_train_path = data_dir+'train/Masks'
    config.train_db_name = 'lmdb_trainset'
    
    # testset
    config.input_test_path = data_dir+'test/Images'
    config.clean_test_path = data_dir+'test/Masks'
    config.test_db_name = 'lmdb_testset'
    
    #validationset
    config.input_val_path = data_dir+'val/Images'
    config.clean_val_path = data_dir+'val/Masks'
    config.val_db_name = 'lmdb_valset'
    
    config.batch_size = batch_size
    config.workers = 5
    
    config.lmdbdir = data_dir+'lmdb/'
    
    train_dataloader = get_train_dataloader(config,transform,image_size,num_classes,gt_type)
    test_dataloader = get_test_dataloader(config,transform,image_size,num_classes,gt_type)
    val_dataloader = get_val_dataloader(config,transform,image_size,num_classes,gt_type)
    
    dataLoader = {'train':train_dataloader,'test':test_dataloader,'val':val_dataloader}

    return dataLoader
    