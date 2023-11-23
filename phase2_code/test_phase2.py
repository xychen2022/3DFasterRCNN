import os
import h5py
import numpy as np
import SimpleITK as sitk
from tensorflow.keras.engine import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers.core import Activation
from tensorflow.keras.layers.convolutional import Conv3D, UpSampling3D
from tensorflow.keras.layers.pooling import MaxPooling3D
from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras import backend as K
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class GroupNormalization(KL.Layer):
    
    def __init__(self,
                 groups=1,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
    
    def build(self, input_shape):
        dim = input_shape[self.axis]
        
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
    
        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')
        
        if dim % self.groups != 0:
            raise ValueError('Number of channels must be divisible by the number of groups')
        
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)
        
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        if self.axis == -1 or self.axis == 4:
            inputs = K.permute_dimensions(inputs, [0, 4, 1, 2, 3])
        
        input_shape = K.int_shape(inputs)  
        original_shape = tf.shape(inputs) #[tf.shape(inputs)[0],] + list(input_shape[1:])
        
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[1] = input_shape[1]

        reshape_group_shape = [tf.shape(inputs)[i] for i in range(len(input_shape))]
        reshape_group_shape[1] = tf.shape(inputs)[1] // self.groups
        group_shape = [tf.shape(inputs)[0], self.groups]
        group_shape.extend(reshape_group_shape[1:])
        
        inputs = K.reshape(inputs, group_shape)
        
        mean = K.mean(inputs, axis=[2, 3, 4, 5], keepdims=True)
        variance = K.var(inputs, axis=[2, 3, 4, 5], keepdims=True)
        
        outputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        outputs = K.reshape(outputs, original_shape)
            
        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma
            
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        if self.axis == -1 or self.axis == 4:
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 4, 1])
        
        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
            }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Head_block:
    def __init__(self, numofbranch, outchannel_per_branch, name='head'):
        self.numofbranch = numofbranch
        self.outchannel_per_branch = outchannel_per_branch
        self.name = name
        assert self.numofbranch == 2
    
    def __call__(self, input1, input2):
        
        conv1 = Conv3D(self.outchannel_per_branch, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv1')(input1)
        gn1 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn1')(conv1)
        
        conv2 = Conv3D(self.outchannel_per_branch, (5, 5, 5), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv2_1')(input2)
        gn2 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2_1')(conv2)
        activ2 = Activation('relu', name=self.name + '_activ2')(gn2)
        conv2 = Conv3D(self.outchannel_per_branch, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv2_2')(activ2)
        gn2 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2_2')(conv2)
        
        out = Concatenate(axis=1, name=self.name + '_concat')([gn1, gn2])
        out = Activation('relu', name=self.name + '_activ_all')(out)
        
        return out

def get_model_heatmap_deep_groupnorm(outputchannel, imagesize, inputchannel):
    
    input1 = Input((1, imagesize[0], imagesize[1], imagesize[2]), name='input_1')
    input2 = Input((1, 2*imagesize[0], 2*imagesize[1], 2*imagesize[2]), name='input_2')
    inputs = Head_block(2, 16)(input1, input2)
    
    conv1 = Conv3D(64, (3, 3, 3), padding='same', data_format='channels_first')(inputs)
    conv1 = GroupNormalization(groups=64, axis=1)(conv1)
    conv1 = Activation('relu')(conv1)
    
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_first')(conv1)
    
    conv2 = Conv3D(64, (3, 3, 3), padding='same', data_format='channels_first')(pool1)
    conv2 = GroupNormalization(groups=64, axis=1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(64, (3, 3, 3), padding='same', data_format='channels_first')(conv2)
    conv2 = GroupNormalization(groups=64, axis=1)(conv2)
    conv2 = Activation('relu')(conv2)
    
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_first')(conv2)
    
    conv3 = Conv3D(128, (3, 3, 3),  padding='same', data_format='channels_first')(pool2)
    conv3 = GroupNormalization(groups=128, axis=1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(128, (3, 3, 3),  padding='same', data_format='channels_first')(conv3)
    conv3 = GroupNormalization(groups=128, axis=1)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_first')(conv3)
    
    conv4 = Conv3D(128, (3, 3, 3), padding='same', data_format='channels_first')(pool3)
    conv4 = GroupNormalization(groups=128, axis=1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(128, (3, 3, 3), padding='same', data_format='channels_first')(conv4)
    conv4 = GroupNormalization(groups=128, axis=1)(conv4)
    conv4 = Activation('relu')(conv4)

    up5 = concatenate([UpSampling3D(size=(2, 2, 2), data_format='channels_first')(conv4), conv3], axis=1)
    conv5 = Conv3D(128, (3, 3, 3), padding='same', data_format='channels_first')(up5)
    conv5 = GroupNormalization(groups=128, axis=1)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(128, (3, 3, 3), padding='same', data_format='channels_first')(conv5)
    conv5 = GroupNormalization(groups=128, axis=1)(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2), data_format='channels_first')(conv5), conv2], axis=1)
    conv6 = Conv3D(128, (3, 3, 3), padding='same', data_format='channels_first')(up6)
    conv6 = GroupNormalization(groups=128, axis=1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(128, (3, 3, 3), padding='same', data_format='channels_first')(conv6)
    conv6 = GroupNormalization(groups=128, axis=1)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2), data_format='channels_first')(conv6), conv1], axis=1)
    conv7 = Conv3D(128, (3, 3, 3), padding='same', data_format='channels_first')(up7)
    conv7 = GroupNormalization(groups=128, axis=1)(conv7)
    conv7 = Activation('relu')(conv7)
    
    conv8 = Conv3D(outputchannel, (3, 3, 3), padding='same', data_format='channels_first')(conv7)
    
    model = Model(inputs=[input1, input2], outputs=conv8)
    model.compile(optimizer='Adam', loss=custom_mse)
    return model

############################################################
#  Loss Functions
############################################################

def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def test():
    
    model.load_weights('checkpoints/Unet.best.hd5')

    size_ = 96
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 4 # very important
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    
    save_path = './predicted_heatmaps'
    
    subject_list = ['case_101_ct_patient', 'case_111_ct_patient', 'case_131_ct_normal', 'case_135_ct_normal', 'case_148_ct_normal', 
                    'case_159_ct_normal', 'case_169_ct_normal', 'case_179_ct_normal', 'case_17_cbct_patient', 'case_32_cbct_patient', 
                    'case_38_cbct_patient', 'case_47_cbct_patient', 'case_54_cbct_patient', 'case_64_cbct_patient', 'case_66_cbct_patient']
    
    for idx in range(len(subject_list)):
        subject_index = subject_list[idx]

        ###########################################################################
        ### Landmarks ###
        with h5py.File(os.path.join(data_path, 'pred_coordinates_phase1/{0}.hdf5'.format(subject_index)), 'r') as f:
            print("Be careful! Is your lmk coordinates in physical space? Or in image space? Also, is your high-resolution image spacing 0.4 mm? If not, change the following accordingly!")
            # locations = f['landmark_i'].value[:, ::-1] # if they denote locations in image space
            locations = f['landmark_i'].value[:, ::-1] / 0.4 # if they denote locations in physical space
        
        locations = locations.astype(np.int32)
        
        present = []
        numLandmarks = locations.shape[0]
        
        for idx in range(numLandmarks):
            landmark_i = locations[idx]
            
            if np.all(landmark_i >= np.array([0, 0, 0])):
                present.append(1)
            else:
                present.append(0)
    
        present = np.array(present)
        locations = locations[np.where(present==1)]
        ###########################################################################
        
        image = sitk.ReadImage(os.path.join(data_path, 'images/subject{0}.nii.gz'.format(subject_index)))
        image = sitk.GetArrayFromImage(image)
        
        image_size = np.array(np.shape(image))
        predicted_heatmaps = np.zeros([outputchannel, image_size[0], image_size[1], image_size[2]], dtype=np.float32)
        count_matrix = np.zeros([outputchannel, image_size[0], image_size[1], image_size[2]], dtype=np.float32)
        
        total_num_of_patches = locations.shape[0]
        center = locations
        
        for idx in range(total_num_of_patches):
            input_patch = np.zeros(patch_size, dtype=np.float32)
            
            subvertex = np.array([center[idx][0]-patch_size[0]//2, center[idx][0]+patch_size[0]//2,
                                  center[idx][1]-patch_size[1]//2, center[idx][1]+patch_size[1]//2,
                                  center[idx][2]-patch_size[2]//2, center[idx][2]+patch_size[2]//2,])
            
            copy_from, copy_to = corrected_crop(subvertex, np.array(image.shape))
            
            cf_z_lower_bound = int(copy_from[0])
            if copy_from[1] is not None:
                cf_z_higher_bound = int(copy_from[1])
            else:
                cf_z_higher_bound = None
            
            cf_y_lower_bound = int(copy_from[2])
            if copy_from[3] is not None:
                cf_y_higher_bound = int(copy_from[3])
            else:
                cf_y_higher_bound = None
            
            cf_x_lower_bound = int(copy_from[4])
            if copy_from[5] is not None:
                cf_x_higher_bound = int(copy_from[5])
            else:
                cf_x_higher_bound = None
        
            input_patch[(copy_to[0]):(copy_to[1]),
                        (copy_to[2]):(copy_to[3]),
                        (copy_to[4]):(copy_to[5])] = \
                        image[cf_z_lower_bound:cf_z_higher_bound,
                              cf_y_lower_bound:cf_y_higher_bound,
                              cf_x_lower_bound:cf_x_higher_bound]
            
            image_one = np.expand_dims(input_patch, axis=0)
            ## Scale 2 ##
            image_two = np.zeros(2*np.array(patch_size), dtype=np.float32)
            
            subvertex = np.array([center[idx][0]-(patch_size[0]//2)*2, center[idx][0]+(patch_size[0]//2)*2,
                                  center[idx][1]-(patch_size[1]//2)*2, center[idx][1]+(patch_size[1]//2)*2,
                                  center[idx][2]-(patch_size[2]//2)*2, center[idx][2]+(patch_size[2]//2)*2])
            
            copy_from2, copy_to2 = corrected_crop(subvertex, np.array(image.shape))
            
            cf_z_lower_bound2 = int(copy_from2[0])
            if copy_from2[1] is not None:
                cf_z_higher_bound2 = int(copy_from2[1])
            else:
                cf_z_higher_bound2 = None
            
            cf_y_lower_bound2 = int(copy_from2[2])
            if copy_from2[3] is not None:
                cf_y_higher_bound2 = int(copy_from2[3])
            else:
                cf_y_higher_bound2 = None
            
            cf_x_lower_bound2 = int(copy_from2[4])
            if copy_from2[5] is not None:
                cf_x_higher_bound2 = int(copy_from2[5])
            else:
                cf_x_higher_bound2 = None

            image_two[(copy_to2[0]):(copy_to2[1]),
                        (copy_to2[2]):(copy_to2[3]),
                        (copy_to2[4]):(copy_to2[5])] = \
                        image[cf_z_lower_bound2:cf_z_higher_bound2,
                              cf_y_lower_bound2:cf_y_higher_bound2,
                              cf_x_lower_bound2:cf_x_higher_bound2]
            
            image_two = np.expand_dims(image_two, axis=0)

            image_1 = np.expand_dims(image_one, axis=0)
            image_2 = np.expand_dims(image_two, axis=0)
            prediction = model.predict([image_1, image_2])
            print(prediction.shape)

            predicted_heatmaps[:,
                               cf_z_lower_bound:cf_z_higher_bound,
                               cf_y_lower_bound:cf_y_higher_bound,
                               cf_x_lower_bound:cf_x_higher_bound] += prediction[0][:,
                                                                                    copy_to[0]:copy_to[1],
                                                                                    copy_to[2]:copy_to[3],
                                                                                    copy_to[4]:copy_to[5]]

            count_matrix[:,
                         cf_z_lower_bound:cf_z_higher_bound,
                         cf_y_lower_bound:cf_y_higher_bound,
                         cf_x_lower_bound:cf_x_higher_bound] += 1.0
        
        heatmaps = predicted_heatmaps/(1.0*count_matrix)
        
        heatmaps = heatmaps[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]]
        
        if not os.path.exists(save_path + '{0}'.format(subject_index)):
            os.makedirs(save_path + '{0}'.format(subject_index))
        
        for idx in range(outputchannel):
            output_image_to_save = sitk.GetImageFromArray(heatmaps[idx].astype(np.float32))
            output_image_to_save.SetSpacing([0.4, 0.4, 0.4])
            sitk.WriteImage(output_image_to_save, save_path + '{0}/pred_{0}_heatmap{1}.nii.gz'.format(subject_index, idx+1))

def corrected_crop(array, image_size):
    array_ = array.copy()
    image_size_ = image_size.copy()
    
    copy_from = [0, 0, 0, 0, 0, 0] #np.zeros([6,], dtype=np.int32)
    copy_to = [0, 0, 0, 0, 0, 0] #np.zeros([6,], dtype=np.int32)
    ## 0 ##
    if array[0] < 0:
        copy_from[0] = 0
        copy_to[0] = int(abs(array_[0]))
    else:
        copy_from[0] = int(array_[0])
        copy_to[0] = 0
    ## 1 ##
    if array[1] > image_size_[0]:
        copy_from[1] = None
        copy_to[1] = -int(array_[1] - image_size_[0])
    else:
        copy_from[1] = int(array_[1])
        copy_to[1] = None
    ## 2 ##
    if array[2] < 0:
        copy_from[2] = 0
        copy_to[2] = int(abs(array_[2]))
    else:
        copy_from[2] = int(array_[2])
        copy_to[2] = 0
    ## 3 ##
    if array[3] > image_size_[1]:
        copy_from[3] = None
        copy_to[3] = -int(array_[3] - image_size_[1])
    else:
        copy_from[3] = int(array_[3])
        copy_to[3] = None
    ## 4 ##
    if array[4] < 0:
        copy_from[4] = 0
        copy_to[4] = int(abs(array_[4]))
    else:
        copy_from[4] = int(array_[4])
        copy_to[4] = 0
    ## 5 ##  
    if array[5] > image_size_[2]:
        copy_from[5] = None
        copy_to[5] = -int(array_[5] - image_size_[2])
    else:
        copy_from[5] = int(array_[5])
        copy_to[5] = None

    return copy_from, copy_to

NUM_LMKS = 18
inputchannel = 1
outputchannel = NUM_LMKS
data_path = "./data"

if __name__ == '__main__':
    model = get_model_heatmap_deep_groupnorm(outputchannel, [96, 96, 96], inputchannel)
    test()
