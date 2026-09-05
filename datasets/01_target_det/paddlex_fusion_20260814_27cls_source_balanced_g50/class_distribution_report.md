# 26 类融合目标检测数据集分布报告

- dataset_dir: `datasets\01_target_det\paddlex_fusion_20260814_27cls_source_balanced_g50`
- weak_threshold: `500`

## Split Summary

| split | images | annotations |
| --- | ---: | ---: |
| train | 10989 | 22558 |
| val | 1400 | 3539 |
| test | 1385 | 3694 |

## Class Counts

| class | total | train | val | weak |
| --- | ---: | ---: | ---: | --- |
| water_l3 | 618 | 450 | 68 | no |
| water_l2 | 563 | 450 | 54 | no |
| water_l1 | 686 | 500 | 95 | no |
| water | 1772 | 1300 | 245 | no |
| order | 1890 | 1502 | 189 | no |
| cylinder_set | 1576 | 1108 | 240 | no |
| cylinder_3 | 886 | 671 | 103 | no |
| cylinder_2 | 912 | 695 | 111 | no |
| cylinder_1 | 926 | 708 | 104 | no |
| ball_yellow | 1106 | 879 | 111 | no |
| ball_blue | 1140 | 911 | 116 | no |
| animal | 3042 | 2421 | 312 | no |
| name | 3974 | 2696 | 563 | no |
| danyuan_2 | 508 | 335 | 53 | no |
| danyuan_1 | 599 | 472 | 59 | no |
| storage | 697 | 497 | 100 | no |
| lable_yellow | 582 | 438 | 94 | no |
| lable_blue | 599 | 450 | 99 | no |
| rape | 916 | 726 | 94 | no |
| broccoli | 938 | 751 | 95 | no |
| potato | 915 | 727 | 93 | no |
| celery | 934 | 739 | 95 | no |
| mushroom | 953 | 763 | 98 | no |
| flammulina velutipes | 949 | 749 | 100 | no |
| tomato | 582 | 450 | 82 | no |
| green bean | 938 | 748 | 98 | no |
| green pepper | 590 | 422 | 68 | no |

## Source Datasets

- `Target_storage0814`: 697 images, 697 annotations
- `Target_shucai0722(2)\2762351_1784723534`: 802 images, 5526 annotations
- `Target_shucai0722(1)\2762349_1784723087`: 2189 images, 2189 annotations
- `My_Formal_Target_kunchong yuan`: 701 images, 2142 annotations
- `My_Formal_Target_0731\2765198_1785427432`: 9385 images, 19237 annotations

## Recommendations

- Do not start formal training before augmented COCO static checks pass.
- Export final model as model.pdmodel + model.pdiparams + infer_cfg.yml for smartcar_baidu_21.
