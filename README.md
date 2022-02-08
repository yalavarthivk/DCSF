# AsyncTSC
This is the source code for the paper (submission id: ) submitted to KDD'22

# Requirements
python                    3.8.11

tensorflow-gpu            2.6.0

tensorflow-datasets       4.2.0

sktime                    0.7.0

pandas                    1.3.2

numpy                     1.19.5

scikit-learn              0.24.2


# Training and Evaluation
1. Physionet dataset

```
python fit_model.py --random_seed 2683894010 --dataset physionet2012 --balance --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/physionet2012/  simple_time --learning_rate 0.00022 --batch_size 128 --warmup_steps 0 --normalize 1 --n_dense_layers 0 --dense_width 512 --dense_dropout 0.2 --n_cnn_layers 1
```

2. MIMIC dataset

```
python fit_model.py --random_seed 952602820 --dataset mimic3_mortality --balance --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/mimic3_mortality/  simple_time --learning_rate 3e-05 --batch_size 64 --warmup_steps 0 --normalize 1 --n_dense_layers 2 --dense_width 64 --dense_dropout 0.3 --n_cnn_layers 3
```

3. Activity dataset

```
python fit_model.py --random_seed 144281024 --dataset activity --max_epochs 1000 --early_stopping 30 --log_dir best_hp_results/activity/  Classifier_RESNET_act_att --learning_rate 0.00084 --batch_size 32 --warmup_steps 0 --normalize 1 --n_dense_layers 3 --dense_width 512 --dense_dropout 0.2 --phi_width 512 --n_cnn_layers 2
```

# Adding datasets

While Physionet dataset can be automatically downloaded, because of the permissions issue you cannot download MIMIC dataset automatically. MIMIC data after preprocessing can be added to ~/Tensorflowdatasets/downloads/manual folder. We have provided the synthetic datsets in Datasets folder. These datasets are also needed to be in ~/Tensorflowdatasets/downloads/manual folder for running the model.
