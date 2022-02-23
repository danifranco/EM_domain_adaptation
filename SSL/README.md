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

        
        As an example those used in our paper are: `Lucchi++`, `VNC` and `Kasthuri++`.

        Which domain is the source and target one, will be specified later.

- Train and evaluate the network:

    * **Step by step**: Using the `.py` files.

        1) Adjust hyperparameter values as well as your file paths in the `Hyperparameters.py` section and run all.
            1.1) It is worth noting that the paths in the hyperparameters file include two options, the regular image path and a secondary path to include the histogram matched files path.
            1.2) If you don't care about histogram_matching you can just specify False in the hm option and ignore this parameter.
        2) Run the SSL_Seg.py from your python environment.

       

    * **Notebook use**: Using the `SSL_notebook.pynb`.

       
