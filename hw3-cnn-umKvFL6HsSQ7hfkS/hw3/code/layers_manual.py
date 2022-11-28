import math

import layers_keras
import numpy as np
import tensorflow as tf

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """
    def call(self, inputs, training=False):
        ## If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        ## Otherwise, manually compute convolution at inference.
        ## Doesn't have to be differentiable. YAY!
        bn, h_in, w_in, c_in = inputs.shape  ## Batch #, height, width, # channels in input
        c_out = self.filters                 ## channels in output
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride

        # Cleaning padding input.
        if self.padding == "SAME":
            h_out = math.ceil(h_in / sh)
            w_out = math.ceil(w_in / sw)

            if h_in % sh == 0:
                ph = max(fh - sh, 0)
            else:
                ph = max(fh - (h_in % sh), 0)
            if w_in % sh == 0:
                pw = max(fw - sw, 0)
            else:
                pw = max(fw - (w_in % sw), 0)

            top = ph // 2
            bottom = ph - top
            left = pw // 2
            right = pw - left

        elif self.padding == "VALID":
            top, bottom, left, right = 0, 0, 0, 0
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")

        ## TODO: Convolve filter from above with the inputs.
        ## Note: Depending on whether you used SAME or VALID padding,
        ## the input and output sizes may not be the same

        ## Pad input if necessary
        padded_inputs = np.pad(inputs, ((0,0), (top,bottom), (left,right), (0,0)))

        ## Calculate correct output dimensions
        h_out = math.floor((h_in + (top + bottom)- fh) / sh + 1)
        w_out = math.floor((w_in + (left + right) - fw) / sw + 1)

        output_array = np.zeros((bn, h_out, w_out, c_out))

        ## Calculate correct output dimensions
        for output_channel in range(c_out):
            for input_channel in range(c_in):
                c_kernel = np.reshape(tf.Variable(self.kernel[:, :, input_channel, output_channel]), (fh, fw))
                for h in range(h_out):
                    for w in range(w_out):
                        region = padded_inputs[:, h*sh:(h*sh + fh), w*sw:(w*sw + fw), input_channel]
                        output_array[:,h,w,output_channel] += np.sum(region * c_kernel, axis=(1,2))
                        


        ## PLEASE RETURN A TENSOR using tf.convert_to_tensor(your_array, dtype=tf.float32)
        return tf.convert_to_tensor(output_array, dtype=tf.float32)