import os
import pickle
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Add
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D, Dense,PReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19

from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

from libs.util import DataLoader, plot_test_images


class SRGAN():
    '''
    Implementation of SRGAN as described in the paper:
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    https://arxiv.org/abs/1609.04802
    '''

    def __init__(self,TPU_train=False,TPU_WORKER=None, height_lr=64, width_lr=64, channels=3, upscaling_factor=4, gen_lr=1e-4, dis_lr=1e-4, gan_lr=1e-4):
        '''
        :param int height_lr: Height of low-resolution images
        :param int width_lr: Width of low-resolution images
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        :param int gan_lr: Learning rate of GAN
        '''

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor % 2 != 0:
            raise ValueError('Upscaling factor must be a multiple of 2; i.e. 2, 4, 8, etc.')
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        # Optimizers used by networks
        optimizer_vgg = Adam(0.0001, 0.9)
        optimizer_generator = Adam(gen_lr, 0.9)
        self.optimizer_discriminator = Adam(dis_lr, 0.9)
        optimizer_gan = Adam(gan_lr, 0.9)
        if TPU_train:
            vggModel = self.build_vgg(optimizer_vgg)
            vggModel.trainable = False
            vggModel.compile(loss='mse',optimizer=optimizer_vgg,metrics=['accuracy'])
            self.vgg = tf.contrib.tpu.keras_to_tpu_model(vggModel,
                                                         strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
            generatorModel = self.build_generator(optimizer_generator,shape_lr=self.shape_lr)
            generatorModel.compile(loss='binary_crossentropy',optimizer=optimizer_generator)
            self.generator = tf.contrib.tpu.keras_to_tpu_model(generatorModel,
                                                         strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
            discriminatorModel = self.build_discriminator(self.optimizer_discriminator)
            discriminatorModel.compile(loss='mse',optimizer=optimizer_discriminator,metrics=['accuracy'])
            self.discriminator = tf.contrib.tpu.keras_to_tpu_model(discriminatorModel,
                                                         strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
            srganModel = self.build_srgan(optimizer_gan)
            srganModel.compile(loss=['binary_crossentropy', 'mse'],loss_weights=[1e-3, 1],optimizer=optimizer_gan)
            self.srgan = tf.contrib.tpu.keras_to_tpu_model(srganModel,
                                                         strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
        else:
        # Build the basic networks
            vggModel = self.build_vgg(optimizer_vgg)
            vggModel.trainable = False
            vggModel.compile(loss='mse',optimizer=optimizer_vgg,metrics=['accuracy'])
            self.vgg = vggModel # model1

            generatorModel = self.build_generator(optimizer_generator,shape_lr=self.shape_lr)
            generatorModel.compile(loss='binary_crossentropy',optimizer=optimizer_generator)
            self.generator = generatorModel # model2

            discriminatorModel = self.build_discriminator(self.optimizer_discriminator)
            discriminatorModel.compile(loss='mse',optimizer=self.optimizer_discriminator,metrics=['accuracy'])
            self.discriminator = discriminatorModel # model3
            # Build the combined network
            srganModel = self.build_srgan(optimizer_gan)
            srganModel.compile(loss=['binary_crossentropy', 'mse'],loss_weights=[1e-3, 1],optimizer=optimizer_gan)
            self.srgan = srganModel

    
    def save_weights(self, filepath):
        """Save the generator and discriminator networks"""
        self.generator.save_weights(filepath + "_generator.h5")
        self.discriminator.save_weights(filepath + "_discriminator.h5")


    def load_weights(self, generator_weights=None, discriminator_weights=None):
        if generator_weights:
            self.generator.load_weights(generator_weights)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights)


    def build_vgg(self, optimizer):
        """
        Load pre-trained VGG19 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        """

        # Input image to extract features from
        img = Input(shape=self.shape_hr)

        # Get the vgg network. Extract features from last conv layer
        vgg = VGG19(include_top=False,weights="imagenet",input_tensor=img,input_shape=self.shape_hr)
        vgg.outputs = [vgg.layers[9].output]

        # Create model and compile
        
        #model.trainable = False
        
        return Model(inputs=img, outputs=vgg.outputs)


    def build_generator(self,optimizer,shape_lr, residual_blocks=16):
        """
        Build the generator network according to description in the paper.
        :param optimizer: Keras optimizer to use for network
        :param int residual_blocks: How many residual blocks to use
        :return: the compiled model
        """

        def residual_block(input):
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x = BatchNormalization(momentum=0.8)(x)
            #x = Activation('relu')(x)
            x = PReLU(alpha_initializer='zeros',shared_axes=[1, 2])(x)
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Add()([x, input])
            return x

        def deconv2d_block(input):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(input)
            x = UpSampling2D(size=2)(x)
            x = PReLU(alpha_initializer='zeros',shared_axes=[1, 2])(x)
            #x = Activation('relu')(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(None,None,3)) #shape_lr

        # Pre-residual
        x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)
        x_start = PReLU(alpha_initializer='zeros',shared_axes=[1, 2])(x_start)
        #x_start = Activation('relu')(x_start)

        # Residual blocks
        r = residual_block(x_start)
        for _ in range(residual_blocks - 1):
            r = residual_block(r)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, x_start])

        # Upsampling (if 4; run twice, if 8; run thrice, etc.)
        for _ in range(int(np.log(self.upscaling_factor) / np.log(2))):
            x = deconv2d_block(x)

        # Generate high resolution output
        hr_output = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        #model = Model(inputs=lr_input, outputs=hr_output)
        return Model(inputs=lr_input, outputs=hr_output)


    def build_discriminator(self, optimizer, filters=64):
        """
        Build the discriminator network according to description in the paper.
        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            
            return d

        # Input high resolution image
        img = Input(shape=self.shape_hr)
        x = conv2d_block(img, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters*2)
        x = conv2d_block(x, filters*2, strides=2)
        x = conv2d_block(x, filters*4)
        x = conv2d_block(x, filters*4, strides=2)
        x = conv2d_block(x, filters*8)
        x = conv2d_block(x, filters*8, strides=2)
        x = Dense(filters*16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Create model and compile
        #model = Model(inputs=img, outputs=x)
        
        return Model(inputs=img, outputs=x)


    def build_srgan(self, optimizer):
        """Create the combined SRGAN network"""

        # Input HR and corresponding LR images
        img_hr = Input(self.shape_hr)
        img_lr = Input(self.shape_lr)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        generated_features = self.vgg(generated_hr)

        # In the combined model we only train the generator
        self.discriminator.trainable = False
        self.discriminator.compile(loss='mse',optimizer=self.optimizer_discriminator,metrics=['accuracy'])
        # Determine whether the generator HR images are OK
        generated_check = self.discriminator(generated_hr)

        # Create model and compile
        #model = Model(inputs=[img_lr, img_hr], outputs=[generated_check, generated_features])
        
        return Model(inputs=[img_lr, img_hr], outputs=[generated_check, generated_features])


    def train(self, epochs, 
        dataname, datapath,
        batch_size=1, 
        test_images=None, test_frequency=50, test_path="./images/samples/", 
        weight_frequency=None, weight_path='./data/weights/', 
        print_frequency=1
    ):
        """Train the SRGAN network
        :param int epochs: how many epochs to train the network for
        :param str dataname: name to use for storing model weights etc.
        :param str datapath: path for the image files to use for training
        :param int batch_size: how large mini-batches to use
        :param list test_images: list of image paths to perform testing on
        :param int test_frequency: how often (in epochs) should testing be performed
        :param str test_path: where should test results be saved
        :param int weight_frequency: how often (in epochs) should network weights be saved. None for never
        :param int weight_path: where should network weights be saved
        :param int print_frequency: how often (in epochs) to print progress to terminal
        """

        # Create data loader
        loader = DataLoader(
            datapath,
            self.height_hr, self.width_hr,
            self.height_lr, self.width_lr,
            self.upscaling_factor
        )

        # Shape of output from discriminator
        disciminator_output_shape = list(self.discriminator.output_shape)
        disciminator_output_shape[0] = batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)

        # VALID / FAKE targets for discriminator
        real = np.ones(disciminator_output_shape)
        fake = np.ones(disciminator_output_shape)        

        # Each epoch == "update iteration" as defined in the paper        
        losses = []
        for epoch in range(epochs):

            # Start epoch time
            if epoch % (print_frequency + 1) == 0:
                start_epoch = datetime.datetime.now()

            # Train discriminator
            imgs_hr, imgs_lr = loader.load_batch(batch_size)
            generated_hr = self.generator.predict(imgs_lr)
            real_loss = self.discriminator.train_on_batch(imgs_hr, real)
            fake_loss = self.discriminator.train_on_batch(generated_hr, fake)
            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)

            # Train generator
            imgs_hr, imgs_lr = loader.load_batch(batch_size)
            features_hr = self.vgg.predict(imgs_hr)
            generator_loss = self.srgan.train_on_batch([imgs_lr, imgs_hr], [real, features_hr])

            # Save losses
            losses.append({'generator': generator_loss, 'discriminator': discriminator_loss})

            # Plot the progress
            if epoch % print_frequency == 0:
                print("Epoch {}/{} | Time: {}s\n>> Generator: {}\n>> Discriminator: {}\n".format(
                    epoch, epochs,
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.3e}".format(k, v) for k, v in zip(self.srgan.metrics_names, generator_loss)]),
                    ", ".join(["{}={:.3e}".format(k, v) for k, v in zip(self.discriminator.metrics_names, discriminator_loss)])
                ))

            # If test images are supplied, show them to the user
            if test_images and epoch % test_frequency == 0:
                plot_test_images(self, loader, test_images, test_path, epoch)

            # Check if we should save the network weights
            if weight_frequency and epoch % weight_frequency == 0:

                # Save the network weights
                self.save_weights(os.path.join(weight_path, dataname))

                # Save the recorded losses
                pickle.dump(losses, open(os.path.join(weight_path, dataname+'_losses.p'), 'wb'))


# Run the SRGAN network
if __name__ == '__main__':

    # Instantiate the SRGAN object
    print(">> Creating the SRGAN network")
    gan = SRGAN(gen_lr=1e-5)

    # Load previous imagenet weights
    print(">> Loading old weights")
    gan.load_weights('../data/weights/imagenet_generator.h5', '../data/weights/imagenet_discriminator.h5')

    # Train the SRGAN
    gan.train(
        epochs=100000,
        dataname='imagenet',
        datapath='../data/imagenet/train/',
        batch_size=16,
        test_images=[
            '../data/buket.jpg'
            
        ],        
        test_frequency=1000,
        test_path='../images/samples/',
        weight_path='../data/weights/',
        weight_frequency=1000,
        print_frequency=10,
)
