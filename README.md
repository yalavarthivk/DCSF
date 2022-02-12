# DCSF
This is the source code for the paper ``DCSF: Deep Convolutional Set Functions for Asynchronous Time Series Classification`` submitted to KDD'22

# Requirements
python                    3.8.11

tensorflow-gpu            2.6.0

tensorflow-datasets       4.2.0

sktime                    0.7.0

pandas                    1.3.2

numpy                     1.19.5

scikit-learn              0.24.2


# Training and Evaluation
1. mini-Physionet dataset

```
python fit_model.py --random_seed 2683894010 --dataset mini_physionet --balance --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/mini_physionet/  dcsf --learning_rate 0.00022 --batch_size 128 --warmup_steps 0 --normalize 1 --n_dense_layers 0 --dense_width 512 --dense_dropout 0.2 --n_cnn_layers 1
```

2. Physionet dataset

```
python fit_model.py --random_seed 34736507 --dataset physionet2012 --balance --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/physionet2012/  dcsf --learning_rate 0.00019 --batch_size 32 --warmup_steps 0 --normalize 1 --n_dense_layers 3 --dense_width 32 --dense_dropout 0.0 --n_cnn_layers 3 
```

3. MIMIC dataset

```
python fit_model.py --random_seed 952602820 --dataset mimic3_mortality --balance --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/mimic3_mortality/  dcsf --learning_rate 3e-05 --batch_size 64 --warmup_steps 0 --normalize 1 --n_dense_layers 2 --dense_width 64 --dense_dropout 0.3 --n_cnn_layers 3
```

4. Activity dataset

```
python fit_model.py --random_seed 144281024 --dataset activity --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/activity/  dcsf_act --learning_rate 0.00084 --batch_size 32 --warmup_steps 0 --normalize 1 --n_dense_layers 3 --dense_width 512 --dense_dropout 0.2 --n_cnn_layers 2
```

5. LSST (async) dataset

```
python fit_model.py --random_seed 3697719380 --dataset LSST_async --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/LSST_async/  dcsf --learning_rate 0.00076 --batch_size 128 --warmup_steps 0 --normalize 0 --n_dense_layers 3 --dense_width 256 --dense_dropout 0.3 --n_cnn_layers 3
```

6. PS (async) dataset

```
python fit_model.py --random_seed 2298336845 --dataset PS_async --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/PS_async/  dcsf --learning_rate 0.0001 --batch_size 32 --warmup_steps 0 --normalize 1 --n_dense_layers 0 --dense_width 32 --dense_dropout 0.2 --n_cnn_layers 2 
```

7. LSST (0.1) dataset

```
python fit_model.py --random_seed 502075857 --dataset LSST_0_1 --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/LSST_0_1/ dcsf --learning_rate 0.00026 --batch_size 32 --warmup_steps 0 --normalize 1 --n_dense_layers 5 --dense_width 512 --dense_dropout 0.0 --n_cnn_layers 3 
```
8. LSST (0.5) dataset

```
python fit_model.py --random_seed 2177597203 --dataset LSST_0_5 --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/LSST_0_5/  dcsf --learning_rate 0.00019 --batch_size 32 --warmup_steps 0 --normalize 0 --n_dense_layers 1 --dense_width 256 --dense_dropout 0.0 --n_cnn_layers 3 

```

9. LSST (0.9) dataset

```
fit_model.py --random_seed 3761597575 --dataset LSST_0_9 --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/LSST_0_9/  dcsf --learning_rate 0.00816 --batch_size 64 --warmup_steps 0 --normalize 1 --n_dense_layers 0 --dense_width 64 --dense_dropout 0.1 --n_cnn_layers 2 
```

10. PhonemeSpectra (0.1) dataset

```
fit_model.py --random_seed 2619880508 --dataset PS_0_1 --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/PS_0_1/  dcsf --learning_rate 0.00038 --batch_size 64 --warmup_steps 0 --normalize 0 --n_dense_layers 4 --dense_width 512 --dense_dropout 0.3 --n_cnn_layers 1
```

11. PhonemeSpectra (0.5) dataset

```
fit_model.py --random_seed 1536644165 --dataset PS_0_5 --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/PS_0_5/  dcsf --learning_rate 8e-05 --batch_size 32 --warmup_steps 0 --normalize 1 --n_dense_layers 3 --dense_width 512 --dense_dropout 0.3 --n_cnn_layers 3 
```

13. PhonemeSpectra (0.9) dataset

```
fit_model.py --random_seed 2301312925 --dataset PS_0_9 --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/PS_0_9/  dcsf --learning_rate 0.0001 --batch_size 128 --warmup_steps 0 --normalize 1 --n_dense_layers 3 --dense_width 512 --dense_dropout 0.2 --n_cnn_layers 2 
```

# Adding datasets

While Physionet dataset can be automatically downloaded, due to the permissions issue, one cannot download MIMIC dataset automatically. MIMIC data after preprocessing can be added to ``~/Tensorflowdatasets/downloads/manual/`` folder. We have provided the synthetic datsets in ``Datasets`` folder. These datasets are also needed to be in ``~/Tensorflowdatasets/downloads/manual/`` folder for running the model.
