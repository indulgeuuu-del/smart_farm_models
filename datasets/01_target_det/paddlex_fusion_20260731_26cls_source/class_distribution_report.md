# 26 类融合目标检测数据集分布报告

- dataset_dir: `datasets\01_target_det\paddlex_fusion_20260731_26cls_source`
- weak_threshold: `700`

## Split Summary

| split | images | annotations |
| --- | ---: | ---: |
| train | 10277 | 21485 |
| val | 1400 | 4112 |
| test | 1400 | 3497 |

## Class Counts

| class | total | train | val | weak |
| --- | ---: | ---: | ---: | --- |
| water_l3 | 618 | 500 | 46 | yes |
| water_l2 | 563 | 400 | 63 | yes |
| water_l1 | 686 | 495 | 91 | yes |
| water | 1772 | 1321 | 238 | no |
| order | 1890 | 1508 | 200 | no |
| cylinder_set | 1576 | 1126 | 222 | no |
| cylinder_3 | 886 | 687 | 131 | no |
| cylinder_2 | 912 | 702 | 136 | no |
| cylinder_1 | 926 | 699 | 147 | no |
| ball_yellow | 1106 | 695 | 184 | no |
| ball_blue | 1140 | 725 | 187 | no |
| animal | 3042 | 2386 | 327 | no |
| name | 3974 | 3016 | 463 | no |
| danyuan_2 | 508 | 489 | 17 | yes |
| danyuan_1 | 599 | 405 | 85 | yes |
| lable_yellow | 582 | 388 | 94 | yes |
| lable_blue | 599 | 393 | 106 | yes |
| rape | 916 | 667 | 166 | no |
| broccoli | 938 | 678 | 174 | no |
| potato | 915 | 665 | 163 | no |
| celery | 934 | 674 | 171 | no |
| mushroom | 953 | 690 | 178 | no |
| flammulina velutipes | 949 | 689 | 171 | no |
| tomato | 582 | 393 | 89 | yes |
| green bean | 938 | 673 | 180 | no |
| green pepper | 590 | 421 | 83 | yes |

## Source Datasets

- `Target_shucai0722(2)\2762351_1784723534`: 802 images, 5526 annotations
- `Target_shucai0722(1)\2762349_1784723087`: 2189 images, 2189 annotations
- `My_Formal_Target_kunchong yuan`: 701 images, 2142 annotations
- `My_Formal_Target_0731\2765198_1785427432`: 9385 images, 19237 annotations

## Recommendations

- Weak classes need train-only extra augmentation: water_l3, water_l2, water_l1, danyuan_2, danyuan_1, lable_yellow, lable_blue, tomato, green pepper
- For water_l1/water_l2/water_l3, keep stronger exposure/shadow/noise augmentation and inspect labels visually.
- Do not start formal training before augmented COCO static checks pass.
- Export final model as model.pdmodel + model.pdiparams + infer_cfg.yml for smartcar_baidu_21.
