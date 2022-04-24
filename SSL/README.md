# Self-supervised Learning U-Net

<img style="display: block;
        margin-left: auto;
        margin-right: auto;
        width: 90%;"
     src="./SSL_diagram.svg">
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
- Clone the repo

```Bash
git clone https://github.com/danifranco/EM_domain_adaptation.git EM_domain_adaptation
cd EM_domain_adaptation/SSL
```
- Install the python libraries provided in the `requirements.txt` file:

```Bash
python3 -m pip install -r requirements.txt
```


- In the repo you'll see the following files:
    - `functions.py` contains all the relevant functions to preprocess the images, train and evaluate the network. 
    - `hyperparameters.py` is the main editable file where you can change the defined hyperparameters between the choices you've got listed aside. 
    - `SSL_Seg.py` is the run file, whenever you've already decided your hyperparameters and defined the paths, you'll run this file.

- Prepare the data.

    - Each data or domain directory must be organized as follows but it accepts both .tif and .png/jpg files:

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

        
        As an example those used in our paper are: `Lucchi++`, `VNC` and `Kasthuri++`.

        Which domain is the source and target one, will be specified later.

- Train and evaluate the network:

    * **Step by step**: Using the `.py` files.

        1) Adjust hyperparameter values as well as your file paths in the `Hyperparameters.py` section and run all.
        
            1) It is worth noting that the paths in the hyperparameters file include two options, the regular image path and a secondary path to include the histogram matched files path.
           2) Please keep in mind that all paths work as a relative path except the images path that you'll need to edit them
        2) Run the SSL_Seg.py from your python environment.
 ```Bash
git clone https://github.com/danifranco/EM_domain_adaptation.git EM_domain_adaptation
cd EM_domain_adaptation/SSL
source tutorial-env/bin/activate
python3 -m pip install -r requirements.txt
python3 SSL_Seg.py
```

       

   * **Notebook use**: Using the `SSL_notebook.pynb`.
        1) Edit the hyperparameters of the first cell both for the pretraining and the training. Remember that selecting no pretraining will load weights from a previous experiment and if you don't have them you'll need to first pretrain it. 
        2) Check all the paths from the second cell match your dataset ones
        3) Execute third cell, all code will run together saving images in the plots folder and writing in the output relevant events and results. It would also be created a txt file with the result and a .csv.

       
