# Reddit Challenge #
김성훈 교수님께서 Facebook TensorflowKR 그룹에 소개해주신 Reddit의 내용을 구현한 저장소입니다.
아직 남은 논문들이 많이 있는데 꾸준히 업데이트 할 예정입니다.
[Facebook 게시글 링크](https://www.facebook.com/groups/TensorFlowKR/permalink/683550341986027/)
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
    - DeepLab V3([paper](https://arxiv.org/abs/1706.05587), [code](https://github.com/bhappy10/reddit_challenge/blob/master/deeplab_v3.py))
    - ENET([paper](https://arxiv.org/abs/1606.02147), [code](https://github.com/bhappy10/reddit_challenge/blob/master/enet.py))

- Generative Adversarial Model
    - GAN([paper](https://arxiv.org/abs/1406.2661), [code](https://github.com/bhappy10/reddit_challenge/blob/master/gan.py))
    - DCGAN([paper](https://arxiv.org/abs/1511.06434), [code](https://github.com/bhappy10/reddit_challenge/blob/master/dcgan.py))
    - DiscoGAN([paper](https://arxiv.org/abs/1703.05192), [code](https://github.com/bhappy10/reddit_challenge/blob/master/discogan.py))

- Object Detection Model
    - (구현중)YOLO V2([paper](https://arxiv.org/abs/1612.08242), [code])

## Usage ##
<pre><code>
# VGG16, ResNet50, FCN8s, FCN16s, FCN32s, UNET, PSPNET, DeepLab_v2, DeepLab_v3, ENET, GAN, DCGAN, DiscoGAN
python main.py --model VGG16 --epoch 30 --batch 8 --learning_rate 0.00001
</code></pre>

## environment ##
- Windows 10
- Tensorflow 1.10
- GTX 1060 3GB

## DataSet ##
- MNIST
- PASCAL VOC 2007([Download](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))
- PASCAL VOC 2012([Download](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))
- ADE Challenge 2016([Download](http://sceneparsing.csail.mit.edu/))


## Reference ##
- VGG
    - [edwith 논문으로 시작하는 딥러닝 - 4가지 CNN 살펴보기](https://www.edwith.org/deeplearningchoi/lecture/15296/)
    - [라온피플 블로그](https://laonple.blog.me/220738560542)

- ResNet
    - [edwith 논문으로 시작하는 딥러닝 - 4가지 CNN 살펴보기](https://www.edwith.org/deeplearningchoi/lecture/15296/)
    - [edwith 논문으로 시작하는 딥러닝 - Residual Network가 왜 잘 되는지 해석해보기](https://www.edwith.org/deeplearningchoi/lecture/15566/)
    - [라온피플 블로그](https://laonple.blog.me/220761052425)
    - [stackoverflow](https://stackoverflow.com/questions/43290192/intuition-on-deep-residual-network)

- FCN
    - [edwith 논문으로 시작하는 딥러닝 - 이미지의 각 픽셀을 분류하는 Semantic Segmentation](https://www.edwith.org/deeplearningchoi/lecture/15554/)
    - [모두연 바이오메디컬랩 논문리뷰](https://modulabs-biomedical.github.io/FCN)
    - [라온피플 블로그](https://laonple.blog.me/220958109081)

- UNET
    - [Kerem Turgutlu 블로그](https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066)
    - [모두연 바이오메디컬랩 논문리뷰](https://modulabs-biomedical.github.io/U_Net)
    - [서재덕님 블로그](https://towardsdatascience.com/@SeoJaeDuk)

- PSPNET
    - [Sheng Hu 블로그](https://medium.com/@steve101777/dense-segmentation-pyramid-scene-parsing-pspnet-753b1cb6097c)

- DeepLab
    - [edwith 논문으로 시작하는 딥러닝 - 이미지의 각 픽셀을 분류하는 Semantic Segmentation](https://www.edwith.org/deeplearningchoi/lecture/15554/)
    - [라온피플 블로그](https://laonple.blog.me/221000648527)
    - [김태오님 PR-045](https://www.youtube.com/watch?v=JiC78rUF4iI&t=1s)

- ENET
    - [Robbert Bormans 블로그](https://medium.com/@robbertbormans/summary-of-enet-a-deep-neural-network-architecture-for-real-time-semantic-segmentation-300a24f31e43)
- GAN
    - [edwith 논문으로 시작하는 딥러닝 - Generative Adversarial Network](https://www.edwith.org/deeplearningchoi/lecture/15846/)
    - [유재준님 블로그](http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html)
    - [유재준님 PR-001](https://www.youtube.com/watch?v=L3hz57whyNw)
    - [이활석님 Github](https://github.com/hwalsuklee/tensorflow-generative-model-collections)
    - [김진중님 Github](https://github.com/golbin/TensorFlow-Tutorials/tree/master/09%20-%20GAN)

- DCGAN
    - [이활석님 Github](https://github.com/hwalsuklee/tensorflow-generative-model-collections)

- DiscoGAN
    - [모두연 논문반 발표자료](http://www.modulabs.co.kr/DeepLAB_Paper/15071)
    - [김태오님 블로그](https://taeoh-kim.github.io/blog/gan%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-image-to-image-translation-pix2pix-cyclegan-discogan/)

- YOLO
    - [edwith 논문으로 시작하는 딥러닝 - Image Detection 방법론: AttentionNet, SSD, YOLO, YOLOv2](https://www.edwith.org/deeplearningchoi/lecture/15579/)
    - [이진원님 PR-023](https://www.youtube.com/watch?v=6fdclSGgeio&t=260s)
    - [sogangori님 블로그](https://m.blog.naver.com/PostView.nhn?blogId=sogangori&logNo=221011203855&proxyReferer=https%3A%2F%2Fwww.google.com%2F)