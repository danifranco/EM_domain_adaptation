# Attention Y-Net

Domain adaptation can also be addressed using multi-task networks. In our paper, we propose a novel architecture that builds on top of the Y-Net [1], but using a stable U-Net with attention gates [2,3] as base architecture.

<img style="display: block;
        margin-left: auto;
        margin-right: auto;
        width: 90%;"
     src="./img/att_ynet.png">
</img>

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
cd EM_domain_adaptation/Attention_Y-Net
```

- Prepare the data.

    - Each data or domain directory must be organized as follows:

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
        * `A_domain`: to store the A-domain dataset.
        * `B_domain`: to store the B-domain dataset.
        * If you want to try more combinations, you can add more.
        
        As an example those used in our paper are: `Lucchi++`, `VNC` and `Kasthuri++`.

        Which domain is the source and target one, will be specified later.

        If you are going to train a single combination, for example, source: A-domain and target: B-domain, target labels (`"y/"` folders) will not be needed. You will only need `test/y` in the `test` section if you want to evaluate the model.

- Train and evaluate the network:

    * **Step by step**: Using the `Att_YNet_Notebook.ipynb`.

        You only need to change variable values in the `Parameters` section and run all:

        1) Specify the path of your data in the `data_path` variable.
        2) Specify from `data_path` the directory name with the source dataset and the directory name with target dataset, in `source` and `target` named variables.
        3) Change as you want the values of the rest of the parameters. Depending on the training step, you should use different ones, as for example the `alpha` value. By default, the last training step parameters are set.
        4) Run all.

    * **All at once**: Using the `Att_YNet.py`.

        You only need to change variable values in the `Parameters` section and run all:

        1) Specify the path of your data in the `dataset_path` variable.
        2) Specify in one list, in the `datasets` variable, the datasets (directory names) you want to use from `dataset_path`.

            **Note:** From this list, all possible combinations will be trained the 3 steps and tested also.

        3) Change as you want the values of the rest of the parameters. Those variables, with a list of 3 values, contain in order the value that will be used in each training step.

            STOP CRITERIA SELECTION:

            * ARA:
                - custom_callback = [False, False, True]
            * Patience using validation set:
                - custom_callback = [False, False, False]
                - patience = [7,6,15]
            * Last epoch:
                - custom_callback = [False, False, False]
                - patience = [7,6,150]

            **Note:** Default values are those presented in the paper. If you want to reproduce paper's results using ARA, just keep these parameters.

        4) Run all.
            ```Bash
            python3 Att_YNet.py
            ```
        This will generate, a `.json` file with all the results, where `matrix per repetition` contains the IOU value obtained in the test set of the target dataset, in the last training step of each repetition. `mean matrix` Contains the mean values among all the repetitions, and `std matrix` the standard deviation. Also, a `errors.log` will be generated with all the errors produced during the execution.

        If you want to see cleaner the execution results of the `.json` you can use the [json2mat](json2mat.ipynb) notebook. Just specify the `.json` file path and run all.

        Finally, the weights of each model will be stored. The names of the weights indicate some information about the model: the training step (weights0, for the first step, weights1 for the second, ...), the repetition (r-0, r-1, ...), folder used as a source dataset (src-Kasthuri++, ...), folder used as a target dataset (trg-VNC, ...) and some training parameters.

## References

```
[1] Roels, J., Hennies, J., Saeys, Y., Philips, W. and Kreshuk, A., 2019, April. Domain adaptive segmentation in volume electron microscopy imaging. In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019) (pp. 1519-1522). IEEE.
 
[2] Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B. and Glocker, B., 2018. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.

[3] Franco-Barranco, D., Mu??oz-Barrutia, A. and Arganda-Carreras, I., 2021. Stable deep neural network architectures for mitochondria segmentation on electron microscopy volumes. Neuroinformatics, pp.1-14.
```
