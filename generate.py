import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Model
import os
import numpy as np
from dataset import Emotion
import yaml
import cv2

NUM_YALE_POSES = 10
class GenParser:
    """
    Class to parse and create inputs based on the parameters in a yaml file.
    """

    # Default parameters to use
    DefaultParams = {
        'mode'        : 'single',
        'constrained' : True,
        'id'          : None,
        'em'          : None,
        'or'          : None,
        'ps'          : None,
        'lt'          : None,
        'id_scale'    : 1.0,
        'id_step'     : 0.1,
        'id_min'      : None,
        'id_max'      : None,
        'em_scale'    : 1.0,
        'em_step'     : 0.1,
        'em_min'      : None,
        'em_max'      : None,
        'or_scale'    : 1.0,
        'or_step'     : 0.1,
        'or_min'      : None,
        'or_max'      : None,
        'ps_scale'    : 1.0,
        'ps_step'     : 0.1,
        'ps_min'      : None,
        'ps_max'      : None,
        'lt_scale'    : 1.0,
        'lt_step'     : 0.1,
        'lt_min'      : None,
        'lt_max'      : None,
        'num_images'  : '1s',
        'fps'         : 30,
        'keyframes'   : None,
    }

    def __init__(self, yaml_path):
        self.yaml_file = open(yaml_path, 'r')

        self.modes = {
            'single'     : self.mode_single
        }

    def __del__(self):
        self.yaml_file.close()


    # Methods for generating inputs by mode

    def mode_single(self, params):
        """
        Generate network inputs for a single image.
        """

        if params['id'] is None:
            params['id'] = 0
        if params['em'] is None:
            params['em'] = 'neutral'
        if params['or'] is None:
            params['or'] = 0
        if params['ps'] is None:
            params['ps'] = 0
        if params['lt'] is None:
            params['lt'] = 0

        
        inputs = {
            'identity': np.empty((1, params['num_ids'])),
            'emotion': np.empty((1, Emotion.length())),
            'orientation': np.empty((1, 2)),
        }

        inputs['identity'][0,:] = self.identity_vector(params['id'], params)
        inputs['emotion'][0,:] = self.emotion_vector(params['em'], params)
        inputs['orientation'][0,:] = self.orientation_vector(params['or'], params)

        return inputs

    # Helper methods

    def num_frames(self, val, params):
        """ Gets the number of frames for a value. """

        if isinstance(val, int):
            return val
        elif isinstance(val, str):
            if val.endswith('s'):
                return int( float(val[:-1]) * params['fps'] )
            else:
                raise RuntimeError("Length '{}' not understood".format(val))
        else:
            raise RuntimeError("Length '{}' not understood".format(val))


    def identity_vector(self, value, params):
        """ Create an identity vector for a provided value. """

        if isinstance(value, str):
            if '+' not in value:
                raise RuntimeError("Identity '{}' not understood".format(value))

            try:
                values = [int(x) for x in value.split('+')]
            except:
                raise RuntimeError("Identity '{}' not understood".format(value))
        elif isinstance(value, int):
            values = [value]
        else:
            raise RuntimeError("Identity '{}' not understood".format(value))

        vec = np.zeros((params['num_ids'],))
        for val in values:
            if val < 0 or params['num_ids'] <= val:
                raise RuntimeError("Identity '{}' invalid".format(val))
            vec[val] += 1.0

        return self.constrain(vec, params['constrained'], params['id_scale'],
                params['id_min'], params['id_max'])


    def emotion_vector(self, value, params):
        """ Create an emotion vector for a provided value. """

        if not isinstance(value, str):
            raise RuntimeError("Emotion '{}' not understood".format(value))

        if '+' in value:
            values = value.split('+')
        else:
            values = [value]

        vec = np.zeros((Emotion.length(),))
        for emotion in values:
            try:
                vec += getattr(Emotion, emotion)
            except AttributeError:
                raise RuntimeError("Emotion '{}' is invalid".format(emotion))

        return self.constrain(vec, params['constrained'], params['em_scale'],
                params['em_min'], params['em_max'])


    def orientation_vector(self, value, params):
        """ Create an orientation vector for a provided value. """

        if isinstance(value, int) or isinstance(value, float):
            value = np.deg2rad(value)
            return np.array([np.sin(value), np.cos(value)])

        elif isinstance(value, str):
            if params['constrained']:
                raise RuntimeError("Cannot manually set orientation vector "
                                   "values when constrained is set to True")

            values = value.split()
            if len(values) != 2:
                raise RuntimeError("Orientation '{}' not understood".format(value))

            vec = np.empty((2,))
            try:
                vec[0] = float(values[0])
                vec[1] = float(values[1])
            except ValueError:
                raise RuntimeError("Orientation '{}' not understood".format(value))

            return vec
        else:
            raise RuntimeError("Orientation '{}' not understood".format(value))


    def pose_vector(self, value, params):
        """ Create an pose vector for a provided value. """

        if isinstance(value, str):
            if '+' not in value:
                raise RuntimeError("Pose '{}' not understood".format(value))

            try:
                values = [int(x) for x in value.split('+')]
            except:
                raise RuntimeError("Pose '{}' not understood".format(value))
        elif isinstance(value, int):
            values = [value]
        else:
            raise RuntimeError("Pose '{}' not understood".format(value))

        vec = np.zeros((NUM_YALE_POSES,))
        for val in values:
            if val < 0 or NUM_YALE_POSES <= val:
                raise RuntimeError("Pose '{}' invalid".format(val))
            vec[val] += 1.0

        return self.constrain(vec, params['constrained'], params['ps_scale'],
                params['ps_min'], params['ps_max'])


    def lighting_vector(self, value, params):
        """ Create a lighting vector for a provided value. """

        if isinstance(value, int) or isinstance(value, float):
            value = np.deg2rad(value)
            return np.array([np.sin(value), np.cos(value), np.sin(value), np.cos(value)])

        elif isinstance(value, str):

            values = value.split()
            if len(values) != 2:
                raise RuntimeError("Lighting '{}' not understood".format(value))

            vec = np.empty((4,))
            try:
                # First element is azimuth
                vec[0] = np.sin(float(values[0]))
                vec[1] = np.cos(float(values[0]))
                # Second element is elevation
                vec[2] = np.sin(float(values[1]))
                vec[3] = np.cos(float(values[1]))
            except ValueError:
                raise RuntimeError("Lighting '{}' not understood".format(value))

            return vec
        else:
            raise RuntimeError("Lighting '{}' not understood".format(value))


    def constrain(self, vec, constrained, scale, vec_min, vec_max):
        """ Constrains the emotion vector based on params. """

        if constrained:
            vec = vec / np.linalg.norm(vec)

        if scale is not None:
            vec = vec * scale

        if vec_min is not None and vec_max is not None:
            vec = np.clip(vec, vec_min, vec_max)

        return vec


    # Main parsing method

    def parse_params(self):
        """
        Parses the yaml file and creates input vectors to use with the model.
        """

        #self.yaml_file.seek(0)

        yaml_params = yaml.load(self.yaml_file)

        params = GenParser.DefaultParams

        for field in params.keys():
            if field in yaml_params:
                params[field] = yaml_params[field]

        return params

    def gen_inputs(self, params):
        """
        creates input vectors to use with the model.
        """

        fn = None
        try:
            fn = self.modes[ params['mode'] ]
        except KeyError:
            raise RuntimeError("Mode '{}' is invalid".format(params['mode']))

        return fn(params)
def normalize(x, lower, upper):
    """ This is a simple linear normalization for an array to a given bound interval
        Params:
        x: image that needs to be normalized in this case
        lower: (int) the lower limit of the interval
        upper: (int) the upper limit of the interval
        Return:
        x_norm: the normalized image between the specified interval 
    """
    x_max = np.max(x)
    x_min = np.min(x)
    # The slope of the linear normalization
    m = (upper - lower) / (x_max - x_min)
    # Linear function for the normalization
    x_norm = (m * (x - x_min)) + lower

    return x_norm
def generate_from_yaml(yaml_path, weights_path, output_dir,epsilon,use_cuda=True,
        extension='jpg'):
    """
    Generate images based on parameters specified in a yaml file.
    """

    model = Model(epsilon=epsilon,sensitivity=1).double()
    model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda()

    parser = GenParser(yaml_path)
    try:
        params = parser.parse_params()
    except RuntimeError as e:
        print("Error: Unable to parse '{}'. Encountered exception:".format(yaml_path))
        print(e)
        return
    #params['dataset'] = dataset
    params['num_ids'] = 67
    inputs = parser.gen_inputs(params)
    identity = torch.from_numpy(inputs['identity']).double()
    emotion = torch.from_numpy(inputs['emotion']).double()
    orientation = torch.from_numpy(inputs['orientation']).double()
    if use_cuda:
        identity = identity.cuda()
        orientation = orientation.cuda()
        emotion = emotion.cuda()
    emotion_name = params['em']
    id = params['id']
    image = model(identity,orientation,emotion)
    reverse_preprocess = T.Compose([
        T.ToPILImage(),
        np.array,
    ])
    image = image.cpu().detach()
    '''dimensions = []
    dimensions.append(image[0,:,:].numpy())
    dimensions.append(image[1,:,:].numpy())
    dimensions.append(image[2,:,:].numpy())
    image = np.dstack(dimensions)
    image = normalize(image, 0, 255)'''
    image = reverse_preprocess(image)
    #image = np.array(255*np.clip(image,0,1), dtype=np.uint8)
    print(cv2.imwrite(f'{output_dir}/{emotion_name}_{id}.{extension}', image))

generate_from_yaml('C:/Users/becadoprias/Desktop/GNN-facegenerator/GNN-facegenerator-pytorch/single.yaml', 'logs/50_without_sensitivity/best_weights.pth', 'E:/datosmanfred/Slovennian_research/GNN-weights/tests',10)