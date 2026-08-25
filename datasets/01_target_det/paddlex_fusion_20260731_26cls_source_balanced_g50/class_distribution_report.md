# 26 类融合目标检测数据集分布报告

- dataset_dir: `datasets\01_target_det\paddlex_fusion_20260731_26cls_source_balanced_g50`
- weak_threshold: `700`

## Split Summary

| split | images | annotations |
| --- | ---: | ---: |
| train | 10389 | 21841 |
| val | 1350 | 3375 |
| test | 1338 | 3878 |

## Class Counts

| class | total | train | val | weak |
| --- | ---: | ---: | ---: | --- |
| water_l3 | 618 | 450 | 68 | yes |
| water_l2 | 563 | 400 | 104 | yes |
| water_l1 | 686 | 500 | 95 | yes |
| water | 1772 | 1304 | 250 | no |
| order | 1890 | 1502 | 189 | no |
| cylinder_set | 1576 | 1113 | 234 | no |
| cylinder_3 | 886 | 671 | 103 | no |
| cylinder_2 | 912 | 695 | 109 | no |
| cylinder_1 | 926 | 708 | 104 | no |
| ball_yellow | 1106 | 880 | 111 | no |
| ball_blue | 1140 | 911 | 116 | no |
| animal | 3042 | 2212 | 312 | no |
| name | 3974 | 2733 | 451 | no |
| danyuan_2 | 508 | 367 | 54 | yes |
| danyuan_1 | 599 | 440 | 59 | yes |
| lable_yellow | 582 | 438 | 94 | yes |
| lable_blue | 599 | 450 | 99 | yes |
| rape | 916 | 726 | 94 | no |
| broccoli | 938 | 749 | 95 | no |
| potato | 915 | 727 | 93 | no |
| celery | 934 | 712 | 95 | no |
| mushroom | 953 | 761 | 98 | no |
| flammulina velutipes | 949 | 747 | 100 | no |
| tomato | 582 | 450 | 82 | yes |
| green bean | 938 | 746 | 98 | no |
| green pepper | 590 | 449 | 68 | yes |

## Source Datasets

- `D:\A CarCarCar\26th BaiDu\Online Competition\数据集合集\自己采集的\标注好了的\Target_shucai0722(2)\2762351_1784723534`: 802 images, 5526 annotations
- `D:\A CarCarCar\26th BaiDu\Online Competition\数据集合集\自己采集的\标注好了的\Target_shucai0722(1)\2762349_1784723087`: 2189 images, 2189 annotations
- `D:\A CarCarCar\26th BaiDu\Online Competition\数据集合集\自己采集的\标注好了的\My_Formal_Target_kunchong yuan`: 701 images, 2142 annotations
- `D:\A CarCarCar\26th BaiDu\Online Competition\数据集合集\自己采集的\标注好了的\My_Formal_Target_0731\2765198_1785427432`: 9385 images, 19237 annotations

## Recommendations

- Weak classes need train-only extra augmentation: water_l3, water_l2, water_l1, danyuan_2, danyuan_1, lable_yellow, lable_blue, tomato, green pepper
- For water_l1/water_l2/water_l3, keep stronger exposure/shadow/noise augmentation and inspect labels visually.
- Do not start formal training before augmented COCO static checks pass.
- Export final model as model.pdmodel + model.pdiparams + infer_cfg.yml for smartcar_baidu_21.
