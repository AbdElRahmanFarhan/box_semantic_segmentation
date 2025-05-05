# Box Semantic Segmentation

This package contains training Mask R-CNN model for box instance segmentation using Pytorch. 

## Setup

---

**Dataset**

[SCD: A Stacked Carton Dataset for Detection and Segmentation](https://github.com/yancie-yjr/scd.github.io)

**Augmentations**:

- Since the dataset often contains little background.

```python
RandomEqualize(p=0.2)
RandomPosterize(bits=3, p=0.5)
RandomHorizontalFlip(p=0.5))
RandomPerspective(distortion_scale=0.3, p=0.5)
```

**Model**

- Mask R-CNN [model](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn) with ResNet50 FPN backbone from Pytorch. pre-trained on COCO dataset.
- Changed the last output of the classification and segmentation heads.
- Changed the following parameters for Region Proposal Network.

  

| parameter | value |
| --- | --- |
| rpn_fg_iou_thresh | 0.8 |
| rpn_bg_iou_thresh | 0.4 |
| rpn_positive_fraction | 0.6 |
| rpn_nms_thresh | 0.6 |

**Optimizer**:

SGD with momentum and step learning rate scheduler.

| parameter | value |
| --- | --- |
| lr | 0.005 |
| momentum | 0.9 |
| weight decay | 0.0005 |

## Training

---

The model is trained using the aforementioned parameters for 20 epochs using a batch size of 2.

**Training losses**

![W&B Chart 5_5_2025, 3_21_20 PM.png](assets/WB_Chart_5_5_2025_3_21_20_PM.png)

**How to train?**

- Training is done on Cloab using Nvida L4 Tensor Core GPU with 24 GB memory.
- All you need is to copy the dataset [zip](https://drive.google.com/file/d/1YeZ4mg_qZ4dBvKKfgGF8RQcyOMNoMp37/view) file into your google drive and rename it as OSCD.zip
- W&B was used to do hyper-parameters tuning and visualize the training logs in real-time. You need to create an account there before running the script. When running the script it will ask you to provide an API tocken which you will find on your account.

## Evaluation

---

**Sample predictions**

- Purple is the detected mask.
- Blue is the ground truth mask.

![Untitled presentation.png](assets/sample_results.png)

**Evaluation metrics**

For IoU thresholds 0.5-0.95

| Metric | Value |
| --- | --- |
| Mask AP | 0.790 |
| Mask AR | 0.827 |

**How to evaluate?**

- Copy the trained [model](https://drive.google.com/file/d/1-5y4ohBNiYwDpl7cYOksdVqeQF3g1YR7/view?usp=drive_link) inside the directory OSCD/model
- Run the evaluation script.