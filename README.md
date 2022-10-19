# ttda (test-time data augmentation)

download : [resnext101 model](https://drive.google.com/file/d/1GOZyktzFki_lJNyAO_oVL3hwAbnb_ofT/view?usp=sharing)

test cmd:
```
CUDA_VISIBLE_DEVICES=0 python main.py --config config.yaml --data-path ./examples \
    --checkpoint 3k3ud6f6-resnext101_32x16d-ce-best.pth --out-path predicts.csv
```

log info:
```
INFO:__main__:Build model ...
INFO:__main__:Succeed to load weights from 3k3ud6f6-resnext101_32x16d-ce-best.pth
INFO:__main__:Dataset created : ./examples
INFO:__main__:TTDA config loaded : {'method': 'resize_and_centercrop', 'resize_small_size': 576, 'crop_size': [512, 512], 'flip': True, 'fuse': 'tsharpen'}
100%|████████████████████████████████████████████████████████████████████| 10/10 [00:09<00:00,  1.02it/s]
INFO:__main__:Predictions saved to predicts.csv
INFO:__main__:Done!
```