import numpy as np
import os
from glob import glob
from time import time
from custom_jac import custom_test, CustomSaver
import json
import shutil

################################## 
####         Parameters       ####
##################################

# Select GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Move to the "DAMT-Net_repo" respository (subfolder)
os.chdir("./DAMT-Net_repo")

# Dataset path
dataset_path = './mitochondria/' # CHANGE THIS

# Folders inside the dataset_path (all the combinations will be computed)
datasets = ["Lucchi++", "VNC", "Kasthuri++"]# CHANGE THIS
dataset2int = {"Lucchi++":0, "VNC":1, "Kasthuri++":2}# CHANGE THIS

# Folder where results are going to be stored (images, json and models)
save_dir = "./results/"

# Json names
json_name_ARA = save_dir + 'Results_DAMT-Net_x10(ARA).json'
json_name_val = save_dir + 'Results_DAMT-Net_x10(val).json'
json_name_last = save_dir + 'Results_DAMT-Net_x10(last).json'

# Number of times all combinations are going to be computed
repetitions = 10
# Number of epochs
epoch = 60

####################################

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def rm_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def comp(o):
  	return float(os.path.splitext(os.path.basename(o))[0].split("_")[1])

def get_empty_all_results():
    all_results = {}
    for source in datasets:
        all_results[source] = {}
        for target in datasets:
            if source == target:
                continue
            all_results[source][target]={}
            all_results[source][target]["train"]={}
            all_results[source][target]["train"]['time']=[]
            all_results[source][target]["train"]['morphology']=[]
            all_results[source][target]["test"]={}
            all_results[source][target]["test"]["source"]={}
            all_results[source][target]["test"]["target"]={}
            all_results[source][target]["test"]["selected_snap"]=[]
            for set in ['train', 'test']:
                all_results[source][target]["test"]["source"][set]={}
                all_results[source][target]["test"]["target"][set]={}
                for data in ['iou', 'time']:
                    all_results[source][target]["test"]["source"][set][data]=[]
                    all_results[source][target]["test"]["target"][set][data]=[]
    return all_results

create_dir(save_dir)
all_results_ARA = get_empty_all_results()
all_results_last = get_empty_all_results()
all_results_val = get_empty_all_results()
iou_mat_r_ARA = [np.zeros((len(datasets),)*2).tolist() for r in range(repetitions)]
iou_mat_r_last = [np.zeros((len(datasets),)*2).tolist() for r in range(repetitions)]
iou_mat_r_val = [np.zeros((len(datasets),)*2).tolist() for r in range(repetitions)]
for r in range(repetitions):
    i=0
    for source in datasets:
        for target in datasets:
            if source == target:
                continue
            i+=1
            run_name = source + '_s-t_' + target
            print("\n"+"-"*47)
            print( "{:^25}  {}/{}      Rep:{}/{}".format(run_name, str(i), len(datasets)*(len(datasets)-1), r+1, repetitions ))
            print("-"*47 + "\n")
            print("TRAIN\n")
            dst_path = os.path.join(save_dir, "Rep_"+str(r), run_name)
            dst_path = dst_path + "/" if dst_path[-1]!="/" else dst_path
            create_dir(dst_path)

            source_train_len = len(glob(dataset_path + source +"/train/x/*"))
            target_len = len(glob(dataset_path + target +"/train_val_test/x/*"))
            data_len = np.ceil((source_train_len+target_len)/2) # (/2 reason): each step, one image of each domain is used, like batch 2

            steps = epoch * data_len

            train_command = 'python3 main.py ' + \
            '--data-dir-img "' + dataset_path + source +'/train/x" ' + \
            '--data-dir-label "' + dataset_path + source +'/train/y" ' + \
            '--data-list "' + dataset_path + source +'/train/file_list.txt" ' + \
            '--data-dir-val "' + dataset_path + source +'/val/x" ' + \
            '--data-dir-val-label "' + dataset_path + source +'/val/y" ' + \
            '--data-list-val "' + dataset_path + source +'/val/file_list.txt" ' + \
            '--input-size "512,512" ' + \
            '--data-dir-target "' + dataset_path + target +'/train_val_test/x" ' + \
            '--data-dir-target-label "' + dataset_path + target +'/train_val_test/y" ' + \
            '--data-list-target "' + dataset_path + target +'/train_val_test/file_list.txt" ' + \
            '--input-size-target "512,512" ' + \
            '--num-workers 4 ' + \
            '--gpu 0 ' + \
            '--num-steps {} '.format(int(steps)) + \
            '--save-pred-every {} '.format(int(data_len)*2) + \
            '--save-num-images 0 ' + \
            '--snapshot-dir "'+ dst_path +'snapshots/" '

            # Train
            train_start = time()
            try:
                os.system(train_command)
            except:
                print("++++++++ Train ERROR ++++++++++")
                continue
            train_end = time()
            all_results_val[source][target]["train"]['time'].append(train_end-train_start)
            all_results_ARA[source][target]["train"]['time'].append(train_end-train_start)
            all_results_last[source][target]["train"]['time'].append(train_end-train_start)

            print("\nTEST\n")

            # CVbest6500_0.9241285685264168.pth
            # CV_0.pth
            best_snap_files = glob(dst_path + "snapshots/CVbest*.pth")
            best_snap_files = [x for x in best_snap_files if not ("_D.pth" in x or "_D2.pth" in x)]
            snap_files = glob(dst_path + "snapshots/CV_*.pth")
            snap_files = [x for x in snap_files if not ("_D.pth" in x or "_D2.pth" in x)]	
            best_snap_files.sort(key=comp, reverse=True) # best first
            snap_files.sort(key=comp, reverse=True) # last first
            snap = best_snap_files[0] if best_snap_files else snap_files[0] # best snapshot and if there is no one, the last one

            all_results_val[source][target]["test"]["selected_snap"].append(str(snap))
            shutil.copy(snap, dst_path + os.path.basename(snap))# save best epoch
            all_results_last[source][target]["test"]["selected_snap"].append(str(snap_files[0]))
            shutil.copy(snap, dst_path + os.path.basename(snap_files[0]))# save last epoch
            
            ################# VAL
            for domain in [source, target]:

                sets = ['test'] if domain == source else ['train', 'test']

                for set in sets:
                    rm_dir(dst_path + 'testResImg/'+ domain +'/'+ set +'/')
                    set2 = 'train_val' if set=='train' else set # just for input data dir
                    test_command = 'python3 prediction.py ' + \
                    '--data-dir-test "' + dataset_path + domain +'/'+ set2 +'/x" ' + \
                    '--data-dir-test-label "' + dataset_path + domain +'/'+ set2 +'/y" ' + \
                    '--data-list-test "' + dataset_path + domain +'/'+ set2 +'/file_list.txt" ' + \
                    '--test-model-path "'+ snap +'" ' + \
                    '--num-workers 4 ' + \
                    '--gpu 0 ' + \
                    '--test_aug 1 ' + \
                    '--save-dir "'+ dst_path + 'testResImg/'+ domain +'/'+ set2 +'/val/" '
                    test_start = time()
                    try:
                        os.system(test_command)
                    except:
                        print("++++++++ Test ERROR ++++++++++")
                        continue
                    test_time = (time()-test_start)
                    mean_iou = custom_test(dataset_path + domain +'/'+ set2, dst_path +'testResImg/'+ domain + '/'+ set2 +'/val/_iter_0')

                    if domain == source:
                        all_results_val[source][target]["test"]["source"][set]['time'].append(float(test_time))
                        all_results_val[source][target]["test"]["source"][set]['iou'].append(float(mean_iou))
                    else:
                        all_results_val[source][target]["test"]["target"][set]['time'].append(float(test_time))
                        all_results_val[source][target]["test"]["target"][set]['iou'].append(float(mean_iou))

            iou_mat_r_val[r][dataset2int[source]][dataset2int[target]] = all_results_val[source][target]["test"]["target"]['test']['iou'][-1]    

            ################ LAST
            snap = snap_files[0] # last epoch
            for domain in [source, target]:

                sets = ['test'] if domain == source else ['train', 'test']

                for set in sets:
                    rm_dir(dst_path + 'testResImg/'+ domain +'/'+ set +'/')
                    set2 = 'train_val' if set=='train' else set # just for input data dir
                    test_command = 'python3 prediction.py ' + \
                    '--data-dir-test "' + dataset_path + domain +'/'+ set2 +'/x" ' + \
                    '--data-dir-test-label "' + dataset_path + domain +'/'+ set2 +'/y" ' + \
                    '--data-list-test "' + dataset_path + domain +'/'+ set2 +'/file_list.txt" ' + \
                    '--test-model-path "'+ snap +'" ' + \
                    '--num-workers 4 ' + \
                    '--gpu 0 ' + \
                    '--test_aug 1 ' + \
                    '--save-dir "'+ dst_path + 'testResImg/'+ domain +'/'+ set2 +'/last/" ' 
                    test_start = time()
                    try:
                        os.system(test_command)
                    except:
                        print("++++++++ Test ERROR ++++++++++")
                        continue
                    test_time = (time()-test_start)
                    mean_iou = custom_test(dataset_path + domain +'/'+ set2, dst_path +'testResImg/'+ domain + '/'+ set2 +'/last/_iter_0')

                    if domain == source:
                        all_results_last[source][target]["test"]["source"][set]['time'].append(float(test_time))
                        all_results_last[source][target]["test"]["source"][set]['iou'].append(float(mean_iou))
                    else:
                        all_results_last[source][target]["test"]["target"][set]['time'].append(float(test_time))
                        all_results_last[source][target]["test"]["target"][set]['iou'].append(float(mean_iou))
            iou_mat_r_last[r][dataset2int[source]][dataset2int[target]] = all_results_last[source][target]["test"]["target"]['test']['iou'][-1] 
            
            ################ ARA
            snap_files.sort(key=comp, reverse=False) # first epoch first
            cs = CustomSaver(dataset_path, snap_files, source, target, path_save = save_dir+'tmp/')    
            iou = cs.compute_all()
            morphology = cs.get_results()
            all_results_ARA[source][target]["train"]['morphology'].append(morphology)
            iou_mat_r_ARA[r][dataset2int[source]][dataset2int[target]] = float(iou)
            all_results_ARA[source][target]["test"]["target"]['test']['iou'].append(float(iou))

            # get prediction images
            test_command = 'python3 prediction.py ' + \
                    '--data-dir-test "' + dataset_path + target +'/test/x" ' + \
                    '--data-dir-test-label "' + dataset_path + target +'/test/y" ' + \
                    '--data-list-test "' + dataset_path + target +'/test/file_list.txt" ' + \
                    '--test-model-path "'+ morphology['best_model'] +'" ' + \
                    '--num-workers 4 ' + \
                    '--gpu 0 ' + \
                    '--test_aug 1 ' + \
                    '--save-dir "'+ dst_path + 'testResImg/'+ target +'/test/ARA/" ' 
            try:
                os.system(test_command)
            except:
                print("++++++++ Test ERROR ++++++++++")
                continue
  
            rm_dir(dst_path+'snapshots/')

            ################
            # Save results #
            ################ VAL
            ALL = {
                "progression": "Last finished model{:^25}  {}/{} --- Rep: {}/{}".format(run_name, str(i), len(datasets)*(len(datasets)-1), r+1, repetitions ),
                "act_repetition": r,
                "datasets": dataset2int,
                "matrix per repetition": iou_mat_r_val,
                "mean matrix": np.mean(iou_mat_r_val[:r+1], axis=0).tolist(),
                "parameters": {
                                'model_name' : 'DAMT-Net',
                                'batch_size_value' : '1',
                                'epoch' : epoch,
                                'steps' : steps,
                                'repetitions' : repetitions,
                                'ARA' : False,
                            },
                "results": all_results_val,
            }    
            file = open(json_name_val, 'w')
            file.write(json.dumps(ALL, indent = 4, sort_keys=False))
            file.close()    
            ################ LAST
            ALL = {
                "progression": "Last finished model{:^25}  {}/{} --- Rep: {}/{}".format(run_name, str(i), len(datasets)*(len(datasets)-1), r+1, repetitions ),
                "act_repetition": r,
                "datasets": dataset2int,
                "matrix per repetition": iou_mat_r_last,
                "mean matrix": np.mean(iou_mat_r_last[:r+1], axis=0).tolist(),
                "parameters": {
                                'model_name' : 'DAMT-Net',
                                'batch_size_value' : '1',
                                'epoch' : epoch,
                                'steps' : steps,
                                'repetitions' : repetitions,
                                'ARA' : False,
                            },
                "results": all_results_last,
            }    
            file = open(json_name_last, 'w')
            file.write(json.dumps(ALL, indent = 4, sort_keys=False))
            file.close()   
            ################ ARA
            ALL = {
                "progression": "Last finished model{:^25}  {}/{} --- Rep: {}/{}".format(run_name, str(i), len(datasets)*(len(datasets)-1), r+1, repetitions ),
                "act_repetition": r,
                "datasets": dataset2int,
                "matrix per repetition": iou_mat_r_ARA,
                "mean matrix": np.mean(iou_mat_r_ARA[:r+1], axis=0).tolist(),
                "parameters": {
                                'model_name' : 'DAMT-Net',
                                'batch_size_value' : '1',
                                'epoch' : epoch,
                                'steps' : steps,
                                'repetitions' : repetitions,
                                'ARA' : True,
                            },
                "results": all_results_ARA,
            }    
            file = open(json_name_ARA, 'w')
            file.write(json.dumps(ALL, indent = 4, sort_keys=False))
            file.close()   

print("\n----------------- FIN -----------------\n")
