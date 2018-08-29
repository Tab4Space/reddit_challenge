# Reddit Challenge #

## Index ##
- Classification Model
    - VGG16([paper](https://arxiv.org/abs/1505.06798), [code](https://github.com/bhappy10/reddit_challenge/blob/master/vgg16.py))
    - ResNet50([paper](https://arxiv.org/abs/1512.03385), [code](https://github.com/bhappy10/reddit_challenge/blob/master/resnet50.py))

- Semantic Segmentation Model
    - FCN8s([paper](https://arxiv.org/abs/1411.4038), [code](https://github.com/bhappy10/reddit_challenge/blob/master/fcn8s.py))
    - FCN16s([paper](https://arxiv.org/abs/1411.4038), [code](https://github.com/bhappy10/reddit_challenge/blob/master/fcn16s.py))
    - FCN32s([paper](https://arxiv.org/abs/1411.4038), [code](https://github.com/bhappy10/reddit_challenge/blob/master/fcn32s.py))
    - UNET([paper](https://arxiv.org/abs/1505.04597), [code](https://github.com/bhappy10/reddit_challenge/blob/master/unet.py))
    - PSPNET([paper](https://arxiv.org/abs/1612.01105), [code](https://github.com/bhappy10/reddit_challenge/blob/master/pspnet.py))
    - DeepLab V2([paper](https://arxiv.org/abs/1606.00915), [code](https://github.com/bhappy10/reddit_challenge/blob/master/deeplab_v2.py))
    - DeepLab V3([paper], [code])
    - ENET([paper](https://arxiv.org/abs/1606.02147), [code](https://github.com/bhappy10/reddit_challenge/blob/master/enet.py))

- Generative Adversarial Model
    - GAN([paper](https://arxiv.org/abs/1406.2661), [code](https://github.com/bhappy10/reddit_challenge/blob/master/gan.py))
    - DCGAN([paper](https://arxiv.org/abs/1511.06434), [code](https://github.com/bhappy10/reddit_challenge/blob/master/dcgan.py))
    - DiscoGAN([paper](), [code]())

- Object Detection Model
    - YOLO V2([paper](https://arxiv.org/abs/1612.08242), [code])

## Usage ##
<pre><code>python main.py --model vgg16 --epoch 30 --batch 8 --learning_rate 0.00001</code></pre>

## environment ##
- Windows 10
- Tensorflow 1.8
- GTX 1060 3GB

## DataSet ##
- MNIST
- PASCAL VOC 2007([Download](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))
- PASCAL VOC 2012([Download](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))
- ADE Challenge 2016([Download](http://sceneparsing.csail.mit.edu/))


## Reference ##
- Classification Model
- Semantic Segmentation Model
- Generative Adversarial Model
- Object Detection Model