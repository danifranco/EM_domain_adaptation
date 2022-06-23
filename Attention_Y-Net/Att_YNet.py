import json
from Att_YNet_functions import *
#set_seed()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# directory with all datasets
dataset_path = "./mitochondria/"
# dataset names inside the data_path (all combinations will be computed)
datasets = ['Lucchi++', 'VNC', 'Kasthuri++']

out_models_path = './models/'
sys.stderr = Logger(stream=sys.stderr, filename='errors.log')
cc_path = './Models_cc/'

def main(resul_filename = './Results_YNet.json'):
    
    ######################
    ###   Parameters   ###
    ######################
    
    # model
    model_name = 'Att_YNet'
    # Batch size
    batch_size_value = 1
    # Epochs of patience
    patience = [7,6,15]
    #Number of epochs
    epochs = [50,40,100]
    # reconstruction loss weight: MSE * alpha
    alphas = [0.98,1,0]
    # segmentation loss weight: BCE * beta
    betas  = [0.02,0,1]
    # Scheduler
    schedule = ['reduce', 'reduce', 'oneCycle'] # None # oneCycle # reduce
    # Optimizer
    optimizer_name = ['SGD', 'Adam', 'Adam'] # 'SGD' # 'Adam'
    # initial filters
    num_filters = 32
    # learning rate
    lr = [1e-3, 2e-4, 2e-4] 
    # use only source
    just_src = [False,False,True] 
    # freeze
    freeze = [False,False,True] 
    # Until which layer to freeze: 'bottle_neck' (not contained), 'fork' (bottle_neck included)
    freeze_layer = 'bottle_neck' 
    # Apply histogram matching (50% probability)
    HistMatch = [False, True, True]
    # Fine tunning
    FT = [False, True, True]
    # Use Solidity
    custom_callback = [False, False, True] # if True: ignore patience and use custom criteria
    # number of random patches (with number lower than 0, sequential patches will be used)
    n_patches = 1000
    # morphology analisis mode (store plots and csv with info)
    analysis_mode = False

    # Number of times all models will be computed (to obtain mean and std)
    repetitions = 10
    ######################

    iou_mat_r = [np.zeros((len(datasets),)*2).tolist() for _ in range(repetitions)]
    for r in range(repetitions):
        i = 0
        for source in datasets:
            for target in datasets:
                if source == target:
                    continue
                FT_base_path = ['',]+[out_models_path + '/weights{}-r-{}-Att_YNet-src-{}-trg-{}-mse-{}-bce-{}-nf-{}-bs-{}-{}-{}.h5'.format(
                                        t, r, source, target, alphas[t], betas[t], num_filters, batch_size_value, optimizer_name[t], schedule[t] ) for t in range(3)]  
                i+=1
                run_name = source + '_s-t_' + target
                print("\n"+"-"*45)
                print( "{:^25}  {}/{}      Rep:{}/{}".format(run_name, str(i), len(datasets)*(len(datasets)-1), r+1,repetitions ))
                print("-"*45 + "\n")

                for j in range(3):
                    print( "\n-------{:^20}-------\n".format("TRAIN  "+str(j+1)+'/3'  ))

                    model, train_time, history, morphology = train_main( 
                        source,
                        target,       
                        dataset_path + source +'/train',
                        dataset_path + target +'/train',
                        dataset_path + source +'/test',
                        dataset_path + target +'/test',
                        epochs[j],
                        lr[j],
                        alphas[j],
                        betas[j],
                        just_src[j],
                        freeze[j],
                        FT[j],
                        FT_base_path[j],
                        FT_base_path[j+1],
                        h_cuts,
                        v_cuts,
                        batch_size_value,
                        HistMatch[j],
                        model_name,
                        freeze_layer,
                        patience[j],
                        schedule[j],
                        optimizer_name[j],
                        'mse_and_bce',
                        num_filters,
                        [0.1, 0.1, 0.2, 0.2, 0.3],
                        'elu',
                        True,
                        n_patches,
                        custom_callback[j],
                        cc_path,
                        analysis_mode
                        )

                    all_results[source][target]["train"]["time"].append(float(train_time))
                    if custom_callback[j]:
                        all_results[source][target]["train"]["morphology"].append(morphology)

                    # execution detailed info, per epoch
                    exec_hist = {}
                    exec_hist['exec_id'] = 'exec_{}_step_{}_-{}-_rep_{}'.format(str(i), str(j), run_name, str(r))
                    # summarize history for metrics
                    for metric_name in metrics:
                        exec_hist[metric_name] = history.history[metric_name]
                        exec_hist['val_'+metric_name] = history.history['val_'+metric_name]
                    all_results[source][target]["train"]["history"].append(exec_hist)

                    # ### Evaluation
                    print("\n--- TEST\n")
                    for domain in [source, target]:
                        if domain == source:
                            sets = ['test']
                        else:
                            sets = ['train', 'test']

                        for set in sets:
                            mean_iou, mean_mse, test_time = test(model, 
                                os.path.join(dataset_path, domain, set),
                                batch_size_value)

                            print("Test MSE:", mean_mse)
                            print("Test IoU:", mean_iou)
                            if domain == source:
                                all_results[source][target]["test"]["source"][set]["time"].append(float(test_time))
                                all_results[source][target]["test"]["source"][set]["iou"].append(float(mean_iou))
                                all_results[source][target]["test"]["source"][set]["mse"].append(float(mean_mse))
                            else:
                                all_results[source][target]["test"]["target"][set]["time"].append(float(test_time))
                                all_results[source][target]["test"]["target"][set]["iou"].append(float(mean_iou))
                                all_results[source][target]["test"]["target"][set]["mse"].append(float(mean_mse))
                    
                iou_mat_r[r][dataset2int[source]][dataset2int[target]] = all_results[source][target]["test"]["target"]['test']['iou'][-1]

                ALL = {
                    "progression": "Last finished model{:^25}  {}/{} --- Rep: {}/{}".format(run_name, str(i), len(datasets)*(len(datasets)-1), r+1, repetitions ),
                    "act_repetition": r,
                    "datasets": dataset2int,
                    "matrix per repetition": iou_mat_r,
                    "mean matrix": np.mean(iou_mat_r[:r+1], axis=0).tolist(),
                    "std matrix": np.std(iou_mat_r[:r+1], axis=0).tolist(),
                    "parameters": {
                                    'model_name' : model_name,
                                    'batch_size_value' : batch_size_value,
                                    'epochs' : epochs,
                                    'alphas (mse)' : alphas,
                                    'betas (bce)'  : betas,
                                    'num_filters' : num_filters,
                                    'lr' : lr,
                                    'just_src' : just_src,
                                    'freeze' : freeze,
                                    'freeze_layer' : freeze_layer,
                                    'HistMatch' : HistMatch,
                                    'FT' : FT,
                                    'repetitions' : repetitions,
                                    'schedule' : schedule,
                                    'optimizer_name' : optimizer_name, 
                                    'patience' : patience,
                                    'n_patches' : n_patches,
                                    'hm_prob' : 0.5,
                                    'filter_0s': 0.5,
                                    'custom_callback': custom_callback,
                                    'analysis_mode': analysis_mode,
                                },
                    "results": all_results,
                }
                file = open(resul_filename, 'w')
                file.write(json.dumps(ALL, indent = 4, sort_keys=False))
                file.close()

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
            all_results[source][target]["train"]['history']=[]
            all_results[source][target]["train"]['morphology']=[]
            all_results[source][target]["test"]={}
            all_results[source][target]["test"]["source"]={}
            all_results[source][target]["test"]["target"]={}
            for set in ['train', 'test']:
                all_results[source][target]["test"]["source"][set]={}
                all_results[source][target]["test"]["target"][set]={}
                for data in ['iou', 'time', 'mse']:
                    all_results[source][target]["test"]["source"][set][data]=[]
                    all_results[source][target]["test"]["target"][set][data]=[]
    return all_results

if __name__ == '__main__':
    all_results = get_empty_all_results()
    h_cuts =      {"Lucchi++":4, "VNC":2, "Kasthuri++":7}
    v_cuts =      {"Lucchi++":3, "VNC":4, "Kasthuri++":6}
    metrics = [ 'loss', 'img_loss', 'mask_loss', 'mask_new_jaccard' ]
    dataset2int = {}
    for i,d in enumerate(datasets):
        dataset2int[d] = i
    
    create_dir(out_models_path)
    main()
    print("\n----------------- END -----------------\n")