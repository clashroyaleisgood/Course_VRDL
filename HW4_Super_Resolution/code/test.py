import argparse

import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ESPCN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    images = []  # list([img, path])
    paths = []
    if args.folder:
        os.makedirs(os.path.join(args.folder, '3x'), exist_ok=True)

        for name in list(os.listdir(args.folder)):
            if os.path.isdir(os.path.join(args.folder, name)):
                continue
            image = pil_image.open(os.path.join(args.folder, name)).convert('RGB')
            images += [image]
            paths += [
                os.path.join(args.folder, '3x', name.replace('.', '_pred.'))
            ]

    elif args.image_file:
        image = pil_image.open(args.image_file).convert('RGB')
        images += [image]
        paths += [
            os.path.join(args.folder, '3x', args.image_file.replace('.', '_pred.'))
        ]

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
    else:
        print('ERROR')

    for image, path in zip(images, paths):
        if args.infer:
            lr = image
            bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            # bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
            lr, _ = preprocess(lr, device)
            _, ycbcr = preprocess(bicubic, device)
            with torch.no_grad():
                preds = model(lr).clamp(0.0, 1.0)

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            # output.save(args.image_file.replace('.', '_espcn_x{}.'.format(args.scale)))
            output.save(path)

        else:
            hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
            bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

            lr, _ = preprocess(lr, device)
            hr, _ = preprocess(hr, device)
            _, ycbcr = preprocess(bicubic, device)

            with torch.no_grad():
                preds = model(lr).clamp(0.0, 1.0)

            psnr = calc_psnr(hr, preds)
            print('PSNR: {:.2f}'.format(psnr))

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            # output.save(args.image_file.replace('.', '_espcn_x{}.'.format(args.scale)))
