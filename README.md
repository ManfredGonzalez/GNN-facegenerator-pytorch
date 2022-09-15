# GNN-facegenerator-pytorch
 
 This generative neural network for facial de-identification is based on [William L. Croft et al.](https://hal.inria.fr/hal-02520056/document) approach to apply differencial privacy in facial images. This implementation was based on PyTorch.
 
 ## Install requirements (Python version 3.8).
    pip install -U scikit-learn
    pip install numpy opencv-contrib-python tqdm tensorboard tensorboardX
    pip install torch==1.4.0
    pip install torchvision==0.5.0

## Training
This sections shows the format of the dataset to be used in this framework.
   RafD_frontal_536/
                     Rafd090_01_Caucasian_female_angry_frontal.jpg
                     Rafd090_01_Caucasian_female_contemptuous_frontal.jpg
                     Rafd090_01_Caucasian_female_disgusted_frontal.jpg
                     Rafd090_01_Caucasian_female_fearful_frontal.jpg
                     Rafd090_01_Caucasian_female_happy_frontal.jpg
                     Rafd090_01_Caucasian_female_neutral_frontal.jpg
                     Rafd090_01_Caucasian_female_sad_frontal.jpg
                     Rafd090_01_Caucasian_female_surprised_frontal.jpg
                     Rafd090_02_Caucasian_female_angry_frontal.jpg
                     ...
### Training from scratch
   python train.py --p project_name --batch_size 20 --num_epochs 1000 --lr_scheduler False --data_path '/path/to_the/dataset/'
