import argparse
import time

from vgg16 import VGG16
from resnet50 import ResNet50
from fcn8s import FCN8s
from fcn16s import FCN16s
from fcn32s import FCN32s
from unet import UNET
from pspnet import PSPNET
from deeplab_v2 import DeepLab_v2
from deeplab_v3 import DeepLab_v3
from enet import ENET
from gan import GAN
from dcgan import DCGAN
from discogan import DiscoGAN
from yolo2 import YOLO_V2

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True,
                        choices=['VGG16', 'ResNet50', 'FCN8s', 'FCN16s', 'FCN32s',
                                'UNET', 'PSPNET', 'DeepLab_v2', 'DeepLab_v3', 'ENET',
                                'GAN', 'DCGAN', 'DiscoGAN', 'YOLO_V2'])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--learning', type=float, default=1e-4)

    return parser.parse_args()
    

def main():
    args = read_args()
    networks = [VGG16, ResNet50, FCN8s, FCN16s, FCN32s, UNET, PSPNET, DeepLab_v2, DeepLab_v3, ENET, GAN, DCGAN, DiscoGAN, YOLO_V2]
    
    for net in networks:
        if args.model == net.MODEL:
            model = net(epoch=args.epoch,
                        batch=args.batch,
                        learning_rate=args.learning
                        )

    model.build_model()
    print('\nStart training after 5sec....\n')
    time.sleep(5)

    model.train_model()
    print('\nFinish training\n')


if __name__ == '__main__':
    main()