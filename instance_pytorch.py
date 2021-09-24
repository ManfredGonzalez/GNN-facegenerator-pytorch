import torch
import torch.utils.data
import torchvision
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms

# ---- Enum classes for vector descriptions
class Emotion:
    angry         = [1., 0., 0., 0., 0., 0., 0., 0.]
    contemptuous  = [0., 1., 0., 0., 0., 0., 0., 0.]
    disgusted     = [0., 0., 1., 0., 0., 0., 0., 0.]
    fearful       = [0., 0., 0., 1., 0., 0., 0., 0.]
    happy         = [0., 0., 0., 0., 1., 0., 0., 0.]
    neutral       = [0., 0., 0., 0., 0., 1., 0., 0.]
    sad           = [0., 0., 0., 0., 0., 0., 1., 0.]
    surprised     = [0., 0., 0., 0., 0., 0., 0., 1.]

    @classmethod
    def length(cls):
        return len(Emotion.neutral)

# ---- Loading functions

class RaFDDataset(torch.utils.data.Dataset):
    def __init__(self, directory,resize):
        """
        Constructor for a RaFDDataset object.
        Args:
            directory (str): Directory where the data lives.
        """
        self.directory = directory
        self.resize = resize
        self.to_tensor = transforms.ToTensor()

        # A list of all files in the current directory (no kids, only frontal gaze)
        #self.filenames = [x for x in os.listdir(directory)
        #        if 'Kid' not in x and 'frontal' in x]
        self.filenames = [x for x in os.listdir(directory)]

        # The number of times the directory has been read over
        self.num_iterations = 0

        # Count identities and map each identity present to a contiguous value
        identities = list()
        for filename in self.filenames:
            identity = int(filename.split('_')[1])-1 # Identities are 1-indexed
            if identity not in identities:
                identities.append(identity)
        self.identity_map = dict()
        for idx, identity in enumerate(identities):
            self.identity_map[identity] = idx

        self.num_identities = len(self.identity_map)
        self.num_instances = len(self.filenames)

    def __getitem__(self, index):

        image = cv2.imread( os.path.join(self.directory, self.filenames[index]) )
        height, width, d = image.shape
        trim=24 
        top=24
        width = int(width-2*trim)
        height = int(width*self.resize[0]/self.resize[1])

        image = image[trim+top:trim+height,trim:trim+width,:]

        # Resize and fit between 0-1
        #self.image = misc.imresize( self.image, image_size )
        #self.image = self.image.resize( self.image, image_size )
        image = cv2.resize( image, self.resize )

        image = image / 255.0

        # Parse filename to get parameters
        
        items = self.filenames[index].split('_')

        # Represent orientation as sin/cos vector
        angle = np.deg2rad(float(items[0][-3:])-90)
        orientation = np.array([np.sin(angle), np.cos(angle)])

        identity_index = int(items[1])-1 # Identities are 1-indexed

        emotion = np.array(getattr(Emotion, items[4]))

        identity_vec = np.zeros(len(self.identity_map), dtype=np.float32)
        identity_vec[ self.identity_map[identity_index] ] = 1.

        
        return torch.from_numpy(orientation),torch.from_numpy(identity_vec),torch.from_numpy(emotion),self.to_tensor(image)

    def __len__(self):
        return self.num_instances
'''
# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
raFDDataset = RaFDDataset('datasets/RafD_frontal_536',(128,128))

# own DataLoader
data_loader = torch.utils.data.DataLoader(raFDDataset,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=0)

batch = next(iter(data_loader))
orientations, identities, emotions, images = batch

print(identities[1].shape)
print(emotions[1].shape)
print(orientations[1].shape) '''