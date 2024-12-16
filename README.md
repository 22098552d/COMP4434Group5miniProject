## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download the all datasets from [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download), [Baidu Driver](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) or [Kaggle Datasets](https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`.

You can reproduce the paper's experiment results by:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh
```
You can reproduce the our experiment results by:

```bash
bash ./scripts/long_term_forecast/ETT_script/LogisticRegression_ETTh1_unify.sh
bash ./scripts/long_term_forecast/ETT_script/LSTM_ETTh1_unify.sh
bash ./scripts/long_term_forecast/ETT_script/OurModel_ETTh1_unify.sh
bash ./scripts/long_term_forecast/ETT_script/OurModel_ETTh2_unify.sh
bash ./scripts/long_term_forecast/ETT_script/OurModel_ETTm1_unify.sh
bash ./scripts/long_term_forecast/ETT_script/SVM_ETTh1.sh
```
