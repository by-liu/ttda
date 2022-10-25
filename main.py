# testing script
import os.path as osp
import argparse
import logging
import timm
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn.functional as F

from image_folder import get_dataset
import ttda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--data-path', type=str, default="examples")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--out-path', type=str, default="predicts.csv")

    return parser.parse_args()


def load_checkpoint(model_path: str, model: torch.nn.Module) -> None:
    if not osp.exists(model_path):
        raise FileNotFoundError(
            "Model not found : {}".format(model_path)
        )
    checkpoint = torch.load(model_path)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    logger.info("Succeed to load weights from {}".format(model_path))
    if missing_keys:
        logger.warn("Missing keys : {}".format(missing_keys))
    if unexpected_keys:
        logger.warn("Unexpected keys : {}".format(unexpected_keys))


@torch.no_grad()
def inference(model, dataset, config, save_path):
    model.eval()

    fsave = open(save_path, "w")
    fsave.write("image_id,pred_label,drmax,dr0,dr1,dr2,dr3,dr4\n")

    for i, samples in tqdm(enumerate(dataset), total=len(dataset)):
        image, _, image_id = samples
        inputs = ttda.augment(image, config)
        if isinstance(inputs, list):
            outputs = [
                model(torch.from_numpy(x).cuda()) for x in inputs
            ] # [6 x 1, 6x1, ....]
            outputs = torch.cat(outputs, dim=0)  # 6 x 4 
        else:
            inputs = torch.from_numpy(inputs).cuda()
            outputs = model(inputs)
        predicts = F.softmax(outputs, dim=1)
        predicts = ttda.fuse_predicts(predicts, reduce=config.fuse)
        pred_label = torch.argmax(predicts)

        fsave.write((
            f"{osp.splitext(image_id)[0]},{pred_label},{predicts.max():.5f},"
            f"{predicts[0]:.5f},{predicts[1]:.5f},{predicts[2]:.5f},{predicts[3]:.5f},{predicts[4]:.5f}\n"
        ))
    fsave.close()
    logger.info(f"Predictions saved to {save_path}")

def main():
    args = parse_args()

    logger.info("Build model ...")
    model = timm.create_model(
        'ig_resnext101_32x16d',
        num_classes=5,
        pretrained=False,
        drop_rate=0.5
    )
    model.cuda()
    load_checkpoint(args.checkpoint, model)

    dataset = get_dataset(args.data_path, return_id=True)
    logger.info(f"Dataset created : {args.data_path}")

    config = OmegaConf.load(args.config)
    logger.info(f"TTDA config loaded : {config}")

    inference(model, dataset, config, args.out_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
