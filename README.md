# CFIW code implementation

This repository contains PyTorch code.

Code and dataset will be uploaded recently.

# Train, val, and predict

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
The training set will be made available later...

## val
Validation can be performed on the validation set dataset [download](https://drive.google.com/file/d/1CppRDY2Rw9BSQHQ29GMHHDj9A6xRNjj0/view?usp=drive_link) using the weights we provided.

## predict
We have provided two images of weld TOFD defects from real industrial scenarios in the 'sample_imgs' folder.

# Model Weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Trained Epochs</th>
<th valign="bottom">Trained Weights</th>
<!-- TABLE BODY -->
<tr><td align="center">CFIW</td>
<td align="center">66</td>
<td align="center"><a href="https://drive.google.com/file/d/1HWqTLearxOiwJRirU_fGwILfu9anNiM2/view?usp=drive_link">download</a></td>
</tr>
</tbody></table>


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Contributing
We actively welcome your pull requests! 

