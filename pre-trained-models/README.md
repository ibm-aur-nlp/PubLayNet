# Pre-trained Faster-RCNN and Mask-RCNN models on PubLayNet

## Training configurations

The configuration (ymal) files of our pre-training settings are provided:  
  - [Faster-RCNN](Faster-RCNN/e2e_faster_rcnn_X-101-64x4d-FPN_1x.ymal)
  - [Mask-RCNN](Mask-RCNN/e2e_mask_rcnn_X-101-64x4d-FPN_1x.ymal)

### Convert pre-trained model for fine-tuning on another target dataset

The category-id to label mapping of the pre-trained model is

| Category id | Label |
| :---: | :--- |
| 0 | Background |
| 1 | Text |
| 2 | Title |
| 3 | List |
| 4 | Table |
| 5 | Figure |

The mapping needs to be converted according to your target dataset before fine-tuning. For example, in the experiment of fine-tuning on SPD dataset in our paper, the category-id to label mapping of the SPD dataset is

| Category id | Label |
| :---: | :--- |
| 0 | Background |
| 1 | Text |
| 2 | List |
| 3 | Table |

To convert the pre-trained models for SPD, run
```
cd <YOUR_CLONE_DIR>/PubLayNet/pre-trained-models
python convert_PubLayNet_model.py \
    --PubLayNet_model {Faster,Mask}-RCNN/model-final.pkl \
    --lookup_table '{0:0, 1:1, 2:3, 3:4}' \
    --output <YOUR_OUTPUT_MODEL_PATH>
```

The `lookup_table` argument controls the link from the category-id of the pre-trained model to that of the target dataset. The key of `lookup_table` is the category-id of the target dataset. The value of `lookup_table` is the category-id of the pre-trained model. If there is a category in your target dataset that does not correspond to any category in the pre-trained model, set the value to `-1` for random initialization.
