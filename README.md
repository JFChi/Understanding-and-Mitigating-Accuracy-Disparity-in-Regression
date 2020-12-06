# Understanding and Mitigating Accuracy Disparity in Regression

Offcial codes for the paper "Understanding and Mitigating Accuracy Disparity in Regression".

## Prerequisites

Our codes in running in Python 3.6, Please install the required packages 

```
pip install -r requirements.txt
```

## Usage

To get the preprocess datasets, please run the following python scripts:

```
python prep_adult.py && python prep_crime.py && python prep_law.py
```

However, there is no need to preprocess in our repo since we already provide the preprocessed datasets in the data folder.

To run CENet in our paper with the coefficient for the adversarial loss 1.0, please run:

```
python main_adult.py --model CENet --mu 1.0
```

To run WassersteinNet in our paper with the coefficient for the adversarial loss 1.0, please run:

```
python main_adult.py --model wmlp --mu 1.0
```

To run the regression net that solely minimize the MSE loss, please run:

```
python main_adult.py --model mlp
```

The usages for other datasets are the same.


## Reference

If you use this code for your research and find it helpful, please cite our paper [Understanding and Mitigating Accuracy Disparity in Regression](https://openreview.net/pdf?id=N9oPAFcuYWX).

## Contact

Please email to [jc6ub@virginia.edu](mailto:jc6ub@virginia.edu) should you have any questions or comments.