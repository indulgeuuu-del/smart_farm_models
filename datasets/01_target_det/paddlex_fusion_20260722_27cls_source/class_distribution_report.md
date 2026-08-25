# 26 类融合目标检测数据集分布报告

- dataset_dir: `datasets\01_target_det\paddlex_fusion_20260722_27cls_source`
- weak_threshold: `500`

## Split Summary

| split | images | annotations |
| --- | ---: | ---: |
| train | 8812 | 15290 |
| val | 1102 | 2619 |
| test | 1104 | 2491 |

## Class Counts

| class | total | train | val | weak |
| --- | ---: | ---: | ---: | --- |
| water_l3 | 281 | 210 | 39 | yes |
| water_l2 | 294 | 230 | 30 | yes |
| water_l1 | 306 | 220 | 41 | yes |
| water | 1193 | 859 | 140 | no |
| order | 1602 | 1153 | 275 | no |
| cylinder_set | 834 | 630 | 83 | no |
| cylinder_3 | 623 | 495 | 64 | no |
| cylinder_2 | 585 | 427 | 74 | no |
| cylinder_1 | 574 | 429 | 67 | no |
| ball_yellow | 322 | 245 | 43 | yes |
| ball_blue | 360 | 259 | 51 | yes |
| animal | 1625 | 1283 | 195 | no |
| name | 1953 | 1503 | 223 | no |
| danyuan_2 | 311 | 239 | 42 | yes |
| danyuan_1 | 314 | 229 | 40 | yes |
| storage | 628 | 438 | 110 | no |
| lable_yellow | 468 | 350 | 58 | yes |
| lable_blue | 412 | 310 | 42 | yes |
| rape | 916 | 681 | 119 | no |
| broccoli | 938 | 683 | 126 | no |
| potato | 915 | 685 | 117 | no |
| celery | 934 | 717 | 119 | no |
| mushroom | 953 | 693 | 126 | no |
| flammulina velutipes | 949 | 727 | 118 | no |
| tomato | 582 | 441 | 61 | no |
| green bean | 938 | 707 | 127 | no |
| green pepper | 590 | 447 | 89 | no |

## Source Datasets

- `MyDataset\My_Formal_Target_0619\2741912_1781882115`: 4500 images, 7620 annotations
- `MyDataset\Target_name\2736433_1781342396`: 684 images, 1953 annotations
- `MyDataset\Target_order and danyuan_0608\2732128_1780914380`: 1335 images, 1604 annotations
- `MyDataset\Target_storage0621\2742707_1782055471`: 1508 images, 1508 annotations
- `D:\A CarCarCar\26th BaiDu\Online Competition\数据集合集\自己采集的\标注好了的\Target_shucai0722(1)\2762349_1784723087`: 2189 images, 2189 annotations
- `D:\A CarCarCar\26th BaiDu\Online Competition\数据集合集\自己采集的\标注好了的\Target_shucai0722(2)\2762351_1784723534`: 802 images, 5526 annotations

## Recommendations

- Weak classes need train-only extra augmentation: water_l3, water_l2, water_l1, ball_yellow, ball_blue, danyuan_2, danyuan_1, lable_yellow, lable_blue
- For water_l1/water_l2/water_l3, keep stronger exposure/shadow/noise augmentation and inspect labels visually.
- Do not start formal training before augmented COCO static checks pass.
- Export final model as model.pdmodel + model.pdiparams + infer_cfg.yml for smartcar_baidu_21.
