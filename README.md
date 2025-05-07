# CFIW code implementation

This repository contains PyTorch code.

Code and dataset have both been uploaded.

# Train, test, and predict

## Train

Run train.sh to train model:

```bash
python main.py \
--model 'CFIW' \
--batch-size 32 \
--epochs 200 \
--output_dir './trained_results/' \
--data-path '/your_dataset_path/'
```

## Train and test dataset

* **Train dataset:** The training dataset [download (☁️ Google Drive)](https://drive.google.com/file/d/1y4GidNc0fb45OAvXW_q_DnDAlJotitRY/view?usp=drive_link "Google drive") of weld TOFD images is provided for model training.
* **Test dataset:** Testing can be performed on the test set dataset [download (☁️ Google Drive)](https://drive.google.com/file/d/1CppRDY2Rw9BSQHQ29GMHHDj9A6xRNjj0/view?usp=drive_link "Google drive") using the trained weights we provided.

## predict

We have provided two images of weld TOFD defects from real industrial scenarios in the 'sample_imgs' folder.

# Model Weights

* The model's **trained weights** on the defect dataset of weld TOFD images are provided as follows:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Trained Epochs</th>
<th valign="bottom">Trained Weights(☁️ Google Drive)</th>
<!-- TABLE BODY -->
<tr><td align="center">CFIW</td>
<td align="center">66</td>
<td align="center"><a href="https://drive.google.com/file/d/1HWqTLearxOiwJRirU_fGwILfu9anNiM2/view?usp=drive_link">download</a></td>
</tr>
</tbody></table>

* The model's **pre-trained weights **on the ImageNet100 dataset are provided as follows:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">Pretrained Weights(☁️ Google Drive)</th>
<!-- TABLE BODY -->
<tr><td align="center">CFIW</td>
<td align="center">ImageNet100</td>
<td align="center"><a href="https://drive.google.com/file/d/1WaCu0zBY1h7xy3cU2laYtTFvfUlJHNKD/view?usp=drive_link">download</a></td>
</tr>
</tbody></table>

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Contributing

We actively welcome your pull requests!
