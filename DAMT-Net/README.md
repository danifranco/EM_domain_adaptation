# Attention Y-Net

Domain adaptation can also be addressed using adversarial networks, as DAMT-Net [1] does.

We are going to use the [original DAMT-Net](https://github.com/Jiajin-Yi/DAMT-Net) repository fork and a custom script to train and test sequentially every combinations, the number of repetitions you want (by default, 10).

## Usage

You can train and use the model with your data following these steps:

- Create a Python3 environment:

```Bash
python3 -m venv tutorial-env
```

- Activate it:

```Bash
source tutorial-env/bin/activate
```

- Install the python libraries provided in the `requirements.txt` file:

```Bash
python3 -m pip install -r requirements.txt
```

- Clone the repo

```Bash
git clone https://github.com/danifranco/EM_domain_adaptation.git EM_domain_adaptation
cd EM_domain_adaptation/DAMT-Net
```

- Prepare the data.

    - Each data directory must be organized as follows:

        ```
        data/
            |-- train/
            |    |-- x/
            |    |      training-0001.tif
            |    |      ...
            |    |-- y/
            |    |      training_groundtruth-0001.tif
            |    |        ...
            |-- test/
            |    |-- x/
            |    |      testing-0001.tif
            |    |      ...
            |    |-- y/
            |    |      testing_groundtruth-0001.tif
            |    |      ...
 
        ```
        Image and labels will be matched by the filename order, so it is important to keep the same order.

    - You will need at least two directories:
        * `trainA`: to store the target dataset.
        * `trainB`: to store the source dataset.
        * (if you want to try more combinations, you can add more)

    - Now, using [prep_EM_data](prep_EM_data.ipynb) notebook, convert the data organization to be used in the following steps. To do that, you only need to specify the directories in the first cell of the notebook and run all.
   
- Train and evaluate the network using the `LongTrainTest_all_DAMT.py`:

    You only need to change variable values in the `Parameters` section and run all:
        
    1) Specify the path of your data in the `dataset_path` variable.
    2) Specify in one list, in the `datasets` variable, the datasets (directory names) you want to use from `dataset_path`. 

        **Note:** from this list all possible combinations will be trained and tested.

    3) Change as you want the values of the rest of the parameters. 

        **Note:** Default values are those presented in the paper. If you want to reproduce paper's results keep the default values.

    4) Run all.
        ```Bash
        python3 LongTrainTest_all_DAMT.py
        ```
    This will generate inside the [DAMT-Net_repo](DAMT-Net_repo) respository (inside `results` folder, by default), three different `.json` file with all the results, where `matrix per repetition` contains the IOU value obtained in the test set of the target dataset, in the last training step of each repetition. `mean matrix` Contains the mean values among all the repetitions, and `std matrix` the standard deviation. Each json is identified by the filename, where:

    - `Results_DAMT-Net_x10(ARA).json`: contain results using ARA stop criteria.
    - `Results_DAMT-Net_x10(val).json`: contain results using the best validation epoch.
    - `Results_DAMT-Net_x10(last).json` contain results using last training epoch.

    If you want to see cleaner the execution results of the `.json` you can use the [json2mat](../Attention_Y-Net/json2mat.ipynb) notebook, from `Attention_Y-Net` folder. Just especify the json path and run all.

    Finally, the weights of each model and predicted masks will be stored.

## References

```
[1] J. Peng, J. Yi, and Z. Yuan, “Unsupervised mitochondria segmentation in EM
images via domain adaptive multi-task learning,” IEEE Journal of Selected Topics
in Signal Processing, vol. 14, no. 6, pp. 1199–1209, 2020.
```
