import re
import numpy as np
from mxnet import nd
from keras.applications.xception import Xception


def trans_name(name):
    name = name.split(':')[0]

    if 'block1_' in name:
        digit = int(name[len('block1_conv')])
        if 'bn' in name:
            name = name.replace('block1_conv{}_bn/'.format(digit),
                                'syncbatchnorm{}_'.format(digit-1))
            name = name.replace('moving', 'running')
            name = name.replace('variance', 'var')
        else:
            name = name.replace('block1_conv{}/kernel'.format(digit),
                                'conv{}_weight'.format(digit-1))
        return name

    if 'sepconv' in name:
        if 'bn' in name:
            a = re.search('sepconv\d_bn/', name)
            if a is not None:
                digit = int(name[a.start(0) + len('speconv')])
                name = name.replace('sepconv{}_bn/'.format(digit), 'syncbatchnorm{}_'.format(digit-1))
                name = name.replace('moving', 'running')
                name = name.replace('variance', 'var')
            else:
                raise Exception('wrong name: {}'.format(name))
        elif 'wise' in name:
            a = re.search('sepconv\d/', name)
            if a is not None:
                digit = int(name[a.start(0) + len('speconv')])
                name = name.replace('sepconv{}/'.format(digit), 'separableconv2d{}_'.format(digit-1))
                name = name.replace('wise', '')
                name = name.replace('kernel', 'weight')
            else:
                raise Exception('wrong name: {}'.format(name))
        else:
            raise Exception('wrong name: {}'.format(name))
        return name
   
    if re.match('conv2d_\d/kernel', name):
        digit = int(name[len('conv2d_')])
        if digit <= 3:
            name = 'block{}_downsample_conv0_weight'.format(digit+1)
        else: # digit == 4
            name = 'block13_downsample_conv0_weight'
        return name
    
    if 'batch_normalization_' in name:
        digit = int(name[len('batch_normalization_')])
        if digit <= 3:
            name = name.replace('batch_normalization_{}/'.format(digit),
                                'block{}_downsample_syncbatchnorm0_'.format(digit+1))
        else: # digit == 4
            name = name.replace('batch_normalization_{}/'.format(digit),
                                'block13_downsample_syncbatchnorm0_')
        name = name.replace('moving', 'running')
        name = name.replace('variance', 'var')
        return name

    if 'predictions/kernel' == name:
        name = 'dense0_weight'
        return name
    elif 'predictions/bias' == name:
        name = 'dense0_bias'
        return name

    raise Exception('wrong name: {}'.format(name))


def trans_weight(weight, name):
    if 'depth' in name:
        return np.transpose(weight, (2, 3, 0, 1))
    elif len(weight.shape) == 4:
        return np.transpose(weight, (3, 2, 0, 1))
    elif len(weight.shape) == 2:
        return np.transpose(weight, (1, 0))
    elif len(weight.shape) == 1:
        return weight
    else:
        raise Exception('wrong weight: {} with shape {}'.format(name, weight.shape))

if __name__ == '__main__':
    model = Xception(include_top=True, weights='imagenet', input_tensor=None)
    weights = dict() 
    # dump all weights (trainable and not) to dict {layer_name: layer_weights} 
    for layer in model.layers: 
        for layer, layer_weights in zip(layer.weights, layer.get_weights()): 
            weights[layer.name] = layer_weights

    new_weights = dict()
    for name in weights:
        new_name = trans_name(name)
        new_weights[new_name] = nd.array(trans_weight(weights[name], name))

    nd.save('xception.params', new_weights)
