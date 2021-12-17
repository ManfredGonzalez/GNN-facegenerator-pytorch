
from model import Model
import torch
import numpy as np
import random
from dataset import Emotion
import torchvision.transforms as T
from tqdm import tqdm
def make_gen():

    
    #weights_path = 'logs/250_paper_b_20_6_uplayers_lr4_4000/best_weights.pth'
    weights_path = 'logs/50_paper_b_20_6_uplayers_lr4_1000/last_weights.pth'
    use_cuda = False
    
    gen = Model(epsilon=50,sensitivity=1).double()
    gen.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    gen.requires_grad_(False)
    gen.eval()

    if use_cuda:
        gen.cuda()
    return gen

def net(gen):

    use_cuda = False

    identity = np.zeros(67, dtype=np.float32)
    identity[random.randint(0, 66)] = 1. # rafd dataset has identities listed in between 0 and 67
    identity = torch.unsqueeze(torch.from_numpy(identity).double(),0)
    emotions_names = ['angry','contemptuous','disgusted','fearful','happy','neutral','sad','surprised']
    emotion = torch.unsqueeze(torch.from_numpy(np.array(getattr(Emotion, random.choice(emotions_names)))).double(),0)
    orientation = torch.unsqueeze(torch.from_numpy(np.array([0.,1.])).double(),0)
    if use_cuda:
        identity.cuda()
        emotion.cuda()
        orientation.cuda()
    image = gen(identity,orientation,emotion)

    return image
def net_rafd(gen,image_name,identity_map):

    use_cuda = False

    image_name = image_name.split('_')

    identity = np.zeros(67, dtype=np.float32)
    #Rafd090_01_Caucasian_female_angry_frontal
    identity[identity_map[int(image_name[1])-1]] = 1. # rafd dataset has identities listed in between 0 and 67
    identity = torch.unsqueeze(torch.from_numpy(identity).double(),0)
    emotions_names = ['angry','contemptuous','disgusted','fearful','happy','neutral','sad','surprised']
    emotion = torch.unsqueeze(torch.from_numpy(np.array(getattr(Emotion, image_name[4]))).double(),0)
    orientation = torch.unsqueeze(torch.from_numpy(np.array([0.,1.])).double(),0)
    if use_cuda:
        identity.cuda()
        emotion.cuda()
        orientation.cuda()
    image = torch.squeeze(gen(identity,orientation,emotion))

    return image

if __name__ == '__main__':
    import os
    import cv2

    dataset_path = 'E:/datosmanfred/Slovennian_research/blaz_experiments/rafd_aligned/rafd-frontal_aligned'
    dataset_save = 'E:/datosmanfred/Slovennian_research/blaz_experiments/rafd-frontal_aligned_net_inferences'
    dataset_filetype = 'JPG'
    dataset_newtype = 'jpg'

    is_rafd = True

    img_names = [i for i in os.listdir(dataset_path)] # change ppm into jpg
    img_names.sort()
    img_paths = [os.path.join(dataset_path, i) for i in img_names]
    save_paths = [os.path.join(dataset_save, i.replace(dataset_filetype, dataset_newtype)) for i in img_names]


    identity_map = dict()

    if is_rafd:
        identities = list()
        for filename in img_names:
            identity = int(filename.split('_')[1])-1 # Identities are 1-indexed
            if identity not in identities:
                identities.append(identity)
        for idx, identity in enumerate(identities):
            identity_map[identity] = idx
    def ensure_dir(d):
        #dd = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
    ensure_dir(dataset_save)

    gen = make_gen()
    
    # go over all files
    for img_name, img_path in zip(img_names, img_paths):
        #img_a_path = os.path.join(path_to_original_images, str(name_a)+'.jpg')
        #img_b_path = os.path.join(img_b_dir, '{:05d}.png'.format(name_b))        
        #img_b_orig_path = os.path.join(path_to_original_images, str(name_b)+'.jpg')
        #img_a_path = os.path.join(path_to_original_images, name_a)
        #img_b_path = os.path.join(img_b_dir, name_b)        
        image = torch.from_numpy(cv2.imread(img_path) / 255.0).flatten()
        if os.path.exists(os.path.join(dataset_save, img_name)):
        #if not os.path.exists(img_a_path) or not os.path.exists(img_b_path): # if any of the pipelines failed to detect faces
            print("File already exists, skipping: ", img_name )
            continue

        #img_a = io.imread(img_a_path)
        #img = cv2.imread(img_path)
        print("Processing: ", img_name)
                    # TODO: here use any face analysis model, to estimate data utility from the image (emotion, etc.)
                    # emotion can be passed within k-same-net function into generator network as the parameter
                    # parameter k denotes k-anonymity factor (number of averaged identities)
        if is_rafd:
            deid_img = net_rafd(gen,img_name,identity_map)
        else:
            deid_img = net(gen)
        reverse_preprocess = T.Compose([
            T.ToPILImage(),
            np.array,
        ])
        deid_img = deid_img.cpu().detach()
        deid_img = reverse_preprocess(deid_img)
        #deid_img = cv2.cvtColor(deid_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(dataset_save, img_name), deid_img)
