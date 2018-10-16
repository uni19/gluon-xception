import numpy as np
import cv2




def test_keras(image):
    from keras.applications.xception import Xception

    model = Xception(include_top=True, weights='imagenet', input_tensor=None)
    out = model.predict(image)
    print('keras :', out.argmax())




def test_mxnet(image):
    import mxnet as mx
    from xception import Xception

    image = np.transpose(image, (0, 3, 1, 2))
    net = Xception(1)
    net.load_parameters('xception.params')
    pred = net(mx.nd.array(image)).asnumpy()
    print('mxnet :', pred.argmax())




if __name__ == '__main__':
    image = cv2.imread('1.jpg')
    image = cv2.resize(image, (299, 299))
    image = image.astype(np.float32)
    image /= 255.
    image.shape = (1,) + image.shape

    test_mxnet(image)
    test_keras(image)
