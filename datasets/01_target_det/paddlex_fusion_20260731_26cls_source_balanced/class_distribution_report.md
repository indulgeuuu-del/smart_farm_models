# 26 类融合目标检测数据集分布报告

- dataset_dir: `datasets\01_target_det\paddlex_fusion_20260731_26cls_source_balanced`
- weak_threshold: `700`

## Split Summary

| split | images | annotations |
| --- | ---: | ---: |
| train | 10292 | 20726 |
| val | 1385 | 4419 |
| test | 1400 | 3949 |

## Class Counts

| class | total | train | val | weak |
| --- | ---: | ---: | ---: | --- |
| water_l3 | 618 | 500 | 46 | yes |
| water_l2 | 563 | 400 | 54 | yes |
| water_l1 | 686 | 500 | 95 | yes |
| water | 1772 | 1304 | 248 | no |
| order | 1890 | 1501 | 200 | no |
| cylinder_set | 1576 | 1130 | 217 | no |
| cylinder_3 | 886 | 734 | 81 | no |
| cylinder_2 | 912 | 766 | 74 | no |
| cylinder_1 | 926 | 777 | 70 | no |
| ball_yellow | 1106 | 669 | 184 | no |
| ball_blue | 1140 | 699 | 187 | no |
| animal | 3042 | 2363 | 353 | no |
| name | 3974 | 2812 | 841 | no |
| danyuan_2 | 508 | 322 | 103 | yes |
| danyuan_1 | 599 | 487 | 85 | yes |
| lable_yellow | 582 | 400 | 94 | yes |
| lable_blue | 599 | 400 | 106 | yes |
| rape | 916 | 582 | 167 | no |
| broccoli | 938 | 591 | 179 | no |
| potato | 915 | 579 | 166 | no |
| celery | 934 | 591 | 178 | no |
| mushroom | 953 | 603 | 177 | no |
| flammulina velutipes | 949 | 603 | 168 | no |
| tomato | 582 | 408 | 85 | yes |
| green bean | 938 | 587 | 178 | no |
| green pepper | 590 | 418 | 83 | yes |

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
