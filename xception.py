import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridSequential, HybridBlock
from mxnet.gluon.contrib.nn import SyncBatchNorm




def _make_separable_conv3(channels, in_channels):
    return SeparableConv2D(channels, in_channels, kernel_size=3, padding=1)




class SeparableConv2D(HybridBlock):
    def __init__(self, channels, in_channels, kernel_size, use_bias=False,
                 strides=(1, 1), padding=(0, 0), dilation=(1, 1), **kwargs):
        super(SeparableConv2D, self).__init__(**kwargs)

        with self.name_scope():
            self.depth = nn.Conv2D(
                channels = in_channels,
                groups = in_channels,
                in_channels = in_channels,
                kernel_size = kernel_size,
                strides = strides,
                padding = padding,
                dilation = dilation,
                use_bias = use_bias,
                prefix = 'depth_'
            )
            self.point = nn.Conv2D(
                channels = channels,
                in_channels = in_channels,
                kernel_size = 1,
                padding = 0,
                use_bias = use_bias,
                prefix = 'point_'
            )

    def hybrid_forward(self, F, x):
        return self.point(self.depth(x))




class XceptionExitModule(HybridBlock):
    def __init__(self, out_channels, mid_channels, in_channels,
                           pre_relu, down, num_dev=1, **kwargs):
        super(XceptionExitModule, self).__init__(**kwargs)
        with self.name_scope():
            self.body = HybridSequential()
            if pre_relu:
                self.body.add(nn.Activation('relu'))
            self.body.add(_make_separable_conv3(mid_channels, in_channels))
            self.body.add(SyncBatchNorm(num_devices=num_dev))
            self.body.add(nn.Activation('relu'))
            self.body.add(_make_separable_conv3(out_channels, mid_channels))
            self.body.add(SyncBatchNorm(num_devices=num_dev))
            if down:
                self.body.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
                self.downsample = HybridSequential(prefix='downsample_')
                with self.downsample.name_scope():
                    self.downsample.add(nn.Conv2D(out_channels, kernel_size=1, strides=2, use_bias=False))
                    self.downsample.add(SyncBatchNorm(num_devices=num_dev))
            else:
                self.downsample = None

    def hybrid_forward(self, F, x):
        if self.downsample:
            return self.downsample(x) + self.body(x)
        else:
            return self.body(x)




class XceptionModule(HybridBlock):
    def __init__(self, channels, in_channels, num_dev=1,
                         pre_relu=True, down=True, **kwargs):
        super(XceptionModule, self).__init__(**kwargs)
        with self.name_scope():
            self.body = HybridSequential(prefix='body_')
            if pre_relu:
                self.body.add(nn.Activation('relu'))
            self.body.add(_make_separable_conv3(channels, in_channels))
            self.body.add(SyncBatchNorm(num_devices=num_dev))
            self.body.add(nn.Activation('relu'))
            self.body.add(_make_separable_conv3(channels, channels))
            self.body.add(SyncBatchNorm(num_devices=num_dev))
            if down:
                self.body.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
                self.downsample = HybridSequential(prefix='downsample_')
                with self.downsample.name_scope():
                    self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=2, use_bias=False))
                    self.downsample.add(SyncBatchNorm(num_devices=num_dev))
            else:
                self.body.add(nn.Activation('relu'))
                self.body.add(_make_separable_conv3(channels, channels))
                self.body.add(SyncBatchNorm(num_devices=num_dev))
                self.downsample = None

    def hybrid_forward(self, F, x):
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.body(x)
        return x + residual




class Xception(HybridBlock):
    def __init__(self, num_dev, num_classes=1000, **kwargs):
        super(Xception, self).__init__(**kwargs)

        with self.name_scope():
            self.features = HybridSequential(prefix='')

            # entry flow
            for i in range(2):
                self.features.add(
                    nn.Conv2D(
                        channels = 32 * (i+1),
                        kernel_size = 3,
                        padding = 0,
                        strides = 2 if i == 0 else 1,
                        use_bias = False
                    )
                )
                self.features.add(SyncBatchNorm(num_devices=num_dev))
                self.features.add(nn.Activation('relu'))

            channels = [64, 128, 256, 728]
            for i in range(len(channels) - 1):
                self.features.add(
                    XceptionModule(
                        channels = channels[i+1],
                        in_channels = channels[i],
                        num_dev = num_dev,
                        pre_relu = (i != 0),
                        down = True,
                        prefix = 'block{}_'.format(i+2)
                    )
                )

            # middle flow
            for i in range(8):
                self.features.add(
                    XceptionModule(
                        channels = 728,
                        in_channels = 728,
                        num_dev = num_dev,
                        pre_relu = True,
                        down = False,
                        prefix = 'block{}_'.format(i+5)
                    )
                )

            # exit flow
            self.features.add(
                XceptionExitModule(
                    out_channels = 1024,
                    mid_channels = 728,
                    in_channels = 728,
                    num_dev = num_dev,
                    pre_relu = True,
                    down = True,
                    prefix = 'block13_'
                )
            )
            self.features.add(
                XceptionExitModule(
                    out_channels = 2048,
                    mid_channels = 1536,
                    in_channels = 1024,
                    num_dev = num_dev,
                    pre_relu = False,
                    down = False,
                    prefix = 'block14_'
                )
            )
            self.features.add(nn.Activation('relu'))

            self.output = HybridSequential(prefix='')
            self.output.add(nn.GlobalAvgPool2D())
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(num_classes))


    def hybrid_forward(self, F, x):
        x = self.features(x)
        return F.softmax(self.output(x))
