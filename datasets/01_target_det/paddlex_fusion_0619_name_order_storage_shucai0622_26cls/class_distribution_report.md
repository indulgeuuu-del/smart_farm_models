# 26 类融合目标检测数据集分布报告

- dataset_dir: `datasets\01_target_det\paddlex_fusion_0619_name_order_storage_shucai0622_26cls`
- weak_threshold: `500`

## Split Summary

| split | images | annotations |
| --- | ---: | ---: |
| train | 8034 | 20172 |
| val | 2009 | 6466 |
| test | 2009 | 6466 |

## Class Counts

| class | total | train | val | weak |
| --- | ---: | ---: | ---: | --- |
| water_l3 | 281 | 214 | 67 | yes |
| water_l2 | 294 | 222 | 72 | yes |
| water_l1 | 306 | 237 | 69 | yes |
| water | 1193 | 915 | 278 | no |
| order | 1602 | 1233 | 369 | no |
| cylinder_set | 834 | 643 | 191 | no |
| cylinder_3 | 623 | 469 | 154 | no |
| cylinder_2 | 585 | 448 | 137 | no |
| cylinder_1 | 574 | 429 | 145 | no |
| ball_yellow | 322 | 249 | 73 | yes |
| ball_blue | 360 | 283 | 77 | yes |
| animal | 1625 | 1275 | 350 | no |
| name | 1953 | 1505 | 448 | no |
| danyuan_2 | 311 | 242 | 69 | yes |
| danyuan_1 | 314 | 240 | 74 | yes |
| storage | 628 | 489 | 139 | no |
| lable_yellow | 468 | 362 | 106 | yes |
| lable_blue | 412 | 318 | 94 | yes |
| rape | 2016 | 1551 | 465 | no |
| broccoli | 1980 | 1515 | 465 | no |
| potato | 1440 | 1020 | 420 | no |
| celery | 1545 | 1117 | 428 | no |
| mushroom | 2016 | 1551 | 465 | no |
| flammulina velutipes | 2013 | 1546 | 467 | no |
| tomato | 1485 | 1062 | 423 | no |
| green bean | 1458 | 1037 | 421 | no |

## Source Datasets

- `MyDataset\My_Formal_Target_0619\2741912_1781882115`: 4500 images, 7620 annotations
- `MyDataset\Target_name\2736433_1781342396`: 684 images, 1953 annotations
- `MyDataset\Target_order and danyuan_0608\2732128_1780914380`: 1335 images, 1604 annotations
- `MyDataset\Target_storage0621\2742707_1782055471`: 1508 images, 1508 annotations
- `MyDataset\Target_shucai0622\2743997_1782142138`: 2016 images, 13953 annotations

## Recommendations

- Weak classes need train-only extra augmentation: water_l3, water_l2, water_l1, ball_yellow, ball_blue, danyuan_2, danyuan_1, lable_yellow, lable_blue
- For h_* vegetable classes, keep flip/vflip/rotate180/object-level direction augmentations.
- For water_l1/water_l2/water_l3, keep stronger exposure/shadow/noise augmentation and inspect labels visually.
- Do not start formal training before augmented COCO static checks pass.
- Export final model as model.pdmodel + model.pdiparams + infer_cfg.yml for smartcar_baidu_21.
