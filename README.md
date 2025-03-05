# CFIW code implementation

This repository contains PyTorch code.

Code and dataset will be uploaded recently.

# Train

Run train.sh to train model:

```bash
python main.py \
--model 'CFIW' \
--batch-size 32 \
--epochs 200 \
--output_dir './trained_results/' \
--data-path '/your_dataset_path/'
```
### Main Results on ImageNet-1K
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbones</th>
<th valign="bottom">Method</th>
<th valign="bottom">Pretrain Epochs</th>
<th valign="bottom">Pretrained Weights</th>
<th valign="bottom">Pretrain Logs</th>
<th valign="bottom">Finetune Logs</th>
<!-- TABLE BODY -->
<tr><td align="center">ViT/B-16</td>
<td align="center">LoMaR</td>
<td align="center">1600</td>
<td align="center"><a href="https://drive.google.com/file/d/160kBTk95xOOCDVKPmxVADWtfqSMzRexW/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1OltaZ1JXVDqkYA72ZjbGRA1QzwAqktsU/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1in72Z5ZPcfYuKnfLcwkIjyBOXXPi4CE7/view?usp=sharing">download</a></td>
</tr>
</tbody></table>




# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Contributing
We actively welcome your pull requests! 

