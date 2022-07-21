from baml4demod import *
import numpy as np # algebra and matrices package
from scipy.io import loadmat, savemat
import torch # Machine learning package
import datetime


print('Starting main')

path_of_run =                       'run/'

bTrain0Load1 =                      0
dSetting =                          {   'snr_dB'     : 18,       # Signal to noise ratio per one RX antenna
                                        'modKey'     : '16QAM' } # modulation key, within 'QPSK' , '16QAM' , '64QAM'
tPercentiles =                      (90, 95, 99, 10, 50, 70, 85)
numPilotsInFrame =                  8
iotUplink =                         IotUplink(dSetting)
FirstLayerDim =                     2  # I and Q
LastLayerDim =                      iotUplink.modulator.order  # num of classes is the total QAM symbols
vLayers =                           [FirstLayerDim, 10, 30, 30, LastLayerDim]
bFirstOrderApprox =                 False # True=FOMAML False=MAML
numMetaTrainingIter =               200
startTime =                         datetime.datetime.now()
numPilotsForMetaTraining =          4
numSymbolsInMetaTrainingTask =      3000
vMetaTrainingTasks =                [4,8,16,24,32,64,128,256]
tasks_batch_size_mtr =              16
lr =                                1e-1
lr_shared =                         1e-3 * tasks_batch_size_mtr
numSgdSteps =                       2
num_target_test_symbols =           4000
numTargetStates =                   50
lr_adapt =                          lr
numMetaAdaptIter =                  200
bDegenerateToFrequentist =          False      # {True = Frequentist ; False = Bayesian}
numNets =                           100 # Ensemble prediction, for meta-training and for meta-test
vConventionalAdaptIter =            [100]

if bTrain0Load1:
    ####### freq ml
    print('Loading hyperparameter xi_untrained from file xi_untrained.pt')
    xi_untrained =                  torch.load(path_of_run+'xi_untrained.pt')
    print('Loading hyperparameter xi_adapt from file v_xi_adapt.pt')
    v_xi_adapt =                    torch.load(path_of_run+'v_xi_adapt.pt')
    ####### bayesian ml
    print('Loading hyperparameter bxi_untrained_params from file bxi_untrained_params.pt')
    bxi_untrained_params =          torch.load(path_of_run+'bxi_untrained_params.pt')
    print('Loading hyperparameter v_bxi_adapt_params from file v_bxi_adapt_params.pt')
    v_bxi_adapt_params =            torch.load(path_of_run+'v_bxi_adapt_params.pt')
else:
    print('Starting to meta train: building meta-data-set and after that meat-training')
    lMeta_dataset = []
    print('Building Meta-Training datasets')
    for tau in range(vMetaTrainingTasks[-1]):   # meta-training tasks
        iotUplink.draw_channel_state()          # new channel state
        lMeta_dataset.append(       iotUplink.step(numSymbolsInMetaTrainingTask,True, False) )
        if (tau % 100 == 0):
            print(f"Building Meta-Training datasets: {tau}/{vMetaTrainingTasks[-1]}")
    print('Done building.')
    print_str = 'mtr: '
    ###################################### Freq. Meta-Train VerB:  #####################################################
    xi_untrained =                      FcReluDnn(vLayers)  # draw by random
    torch.save(xi_untrained,    path_of_run+'xi_untrained.pt')
    print(print_str+'File saved xi_untrained.pt')
    v_xi_last =                       []
    v_xi_adapt =                      []
    v_dMtr_f_losses =                 []
    for kk, numMetaTrainingTasks in enumerate(vMetaTrainingTasks):
        iter_str = f'{kk}/{len(vMetaTrainingTasks)} '
        print(print_str+f"Starting MetaTraining for {numMetaTrainingTasks} meta-training tasks")
        ####### freq ml
        if kk==0:
            net_params_init =           deepcopy(xi_untrained)         # start freshly
        else:
            net_params_init =           deepcopy(xi_adapt)        # continue from the last adopted till now
        xi_last, xi_adapt, dMtr_f_losses =    FreqMamlMetaTrainVerB(
                                                            net_params_init,        # net_init, instance of nnModule
                                                            lMeta_dataset[:numMetaTrainingTasks],
                                                                                 # data_set_meta_train (only first ones)
                                                            lr,                     # lr=0.1
                                                            lr_shared,              # lr_shared=0.1,
                                                            tasks_batch_size_mtr,   # tasks_batch_size_mtr=20,
                                                            numSgdSteps,            # numSgdSteps=1,
                                                            numPilotsForMetaTraining,# numPilotsForMetaTraining=16,
                                                            numMetaTrainingIter,    # numMetaTrainingIter=10000
                                                            bFirstOrderApprox,      # bFirstOrderApprox ,TRUE=FOMAML
                                                            print_str+'freq: '+iter_str)  # print_str
        v_xi_last.append(       xi_last)
        v_xi_adapt.append(      xi_adapt)
        v_dMtr_f_losses.append( dMtr_f_losses)
        file_name = path_of_run + "IotMetaTraining_f" + str(numMetaTrainingTasks) + ".mat"
        print(print_str + 'freq: '+ iter_str  + f'Finished MetaTraining. Saving file ' + file_name)
        savemat(file_name, dMtr_f_losses)
    torch.save(v_xi_adapt,        path_of_run+'v_xi_adapt.pt')
    torch.save(v_xi_last,         path_of_run+'v_xi_last.pt')
    print('Files saved: v_xi_adapt.py, v_xi_last.pt')
    ################################## Bayesian Meta-Train VerB:  #####################################################
    nu_untrained =                      FcReluDnn(vLayers)  # some random init, representing the mean of the rv net
    varrho_untrained =                  FcReluDnn(vLayers)  # some random init, representing the exp(std) of the rv net

    for w in varrho_untrained.parameters():
        if bDegenerateToFrequentist:  # we degenerate bayesian to freq by learning only the mean nu and
            w.requires_grad_(False) # disables them as parameters while creating a graph...
            w -= np.inf   # set all varrho parameters to -inf, which means std=exp(-inf)=0 (not stochastic)
    bxi_untrained_params =         ( list(map(lambda p: p[0], zip(    nu_untrained.parameters() )))    # bxi=bayesian
                                   + list(map(lambda p: p[0], zip(varrho_untrained.parameters() )))  )

    torch.save(bxi_untrained_params,    path_of_run+'bxi_untrained_params.pt')
    v_bxi_last_params =                 []
    v_bxi_adapt_params =                []
    v_dMtr_b_losses =                   []
    for kk, numMetaTrainingTasks in enumerate(vMetaTrainingTasks):
        iter_str = f'{kk}/{len(vMetaTrainingTasks)} '
        if kk==0:
            bnet_params_init =          list(map(lambda p: deepcopy(p[0]) , zip(bxi_untrained_params)))
                                        # start freshly
        else:
            bnet_params_init =          list(map(lambda p: deepcopy(p[0]) , zip(bxi_adapt_params)))
                                        # continue from the last adopted till now
        bxi_last_params, bxi_adapt_params, dMtr_b_losses =  BayesianMamlMetaTrain(
                                                        bnet_params_init,           # bnet_init, instance of nnModule
                                                        lMeta_dataset[:numMetaTrainingTasks], # only first
                                                                                    # data_set_meta_train
                                                        lr,                         # lr
                                                        lr_shared,                  # lr_shared
                                                        tasks_batch_size_mtr,       # tasks_batch_size_mtr
                                                        numSgdSteps,                # numSgdSteps
                                                        numPilotsForMetaTraining,   # numPilotsForMetaTraining
                                                        numMetaTrainingIter,        # numMetaTrainingIter
                                                        numNets,                    # numNets
                                                        bFirstOrderApprox,          # bFirstOrderApprox ,TRUE=FOMAML
                                                        bDegenerateToFrequentist,   # bDegenerateToFrequentist
                                                        print_str+'bayesian: '+iter_str)   # print_str
        v_bxi_last_params.append(      bxi_last_params)
        v_bxi_adapt_params.append(     bxi_adapt_params)
        v_dMtr_b_losses.append(        dMtr_b_losses)
        file_name = path_of_run + "IotMetaTraining_b" + str(numMetaTrainingTasks) + ".mat"
        print(print_str + 'bayesian: ' + iter_str + f'Finished MetaTraining. Saving file ' + file_name)
        savemat(file_name, dMtr_b_losses)
    torch.save(v_bxi_adapt_params,        path_of_run+'v_bxi_adapt_params.pt')
    torch.save(v_bxi_last_params,         path_of_run+'v_bxi_last_params.pt')
    print('Files saved: v_bxi_adapt_params.py, v_bxi_last_params.pt')

mSerOptimalDemodTePer =             np.zeros((len(vMetaTrainingTasks), len(tPercentiles)+1))
mSerLmmse_hDemodTePer =             np.zeros((len(vMetaTrainingTasks), len(tPercentiles)+1))
mSerLmmse_xDemodTePer =             np.zeros((len(vMetaTrainingTasks), len(tPercentiles)+1))
mSerFMamlTePer =                    np.zeros((len(vMetaTrainingTasks), len(tPercentiles)+1))
mSerBMamlTePer =                    np.zeros((len(vMetaTrainingTasks), len(tPercentiles)+1))
mSerConventionalTePer =             np.zeros((len(vMetaTrainingTasks), len(tPercentiles)+1,len(vConventionalAdaptIter)))

mSerOptimalDemodTe =                np.zeros((len(vMetaTrainingTasks), numTargetStates))
mSerLmmse_hDemodTe =                np.zeros((len(vMetaTrainingTasks), numTargetStates)) # LMMSE + Min Distance
mSerLmmse_xDemodTe =                np.zeros((len(vMetaTrainingTasks), numTargetStates)) # LMMSE + Min Distance
mSerFMamlTe =                       np.zeros((len(vMetaTrainingTasks), numTargetStates))
mSerBMamlTe =                       np.zeros((len(vMetaTrainingTasks), numTargetStates))
mSerConventionalTe =                np.zeros((len(vMetaTrainingTasks), numTargetStates    ,len(vConventionalAdaptIter)))

mEcePiledFMamlTe =                  np.zeros((len(vMetaTrainingTasks), numTargetStates))
mEcePiledBMamlTe =                  np.zeros((len(vMetaTrainingTasks), numTargetStates))

l_dataset_x_all_test_targets_concat =         []
l_fnet_soft_out_all_test_targets_concat =     []
l_bnet_soft_out_all_test_targets_concat =     []

print_str = 'mda: '
print(print_str+'Exploring MetaAdapting')
for ss in range(numTargetStates):   # target tasks
    metatest_str =                      f"metatest task {ss}/{numTargetStates}: "
    iotUplink.draw_channel_state()  # new channel state
    data_set_target_task_te =           iotUplink.step(num_target_test_symbols, False, False)  # data
    data_set_target_task_tr =           iotUplink.step(numPilotsInFrame,        True,  False)  # pilots
    for kk, numMetaTrainingTasks in enumerate(vMetaTrainingTasks):
        nummtrtasks_str =               f'num mtr tasks {kk}/{len(vMetaTrainingTasks)}: '
        ###### Benchmarks: 1) LMMSE_h + ML (maximum likelihood,=minimum distance) decoding; 2)  LMMSE_x + ML
        complex_x_tr =                  iotUplink.modulator.modulate(   data_set_target_task_tr.x)
        data_set_equalized_h_te =       EqualizerLmmseH(                data_set_target_task_tr.y,
                                                                        complex_x_tr,
                                                                        data_set_target_task_te.y,
                                                                        data_set_target_task_te.x,
                                                                        0.5 /iotUplink.snr_lin)# noiseVarOverChannelGain
        data_set_equalized_x_te =       EqualizerLmmseX(                data_set_target_task_tr.y,
                                                                        complex_x_tr,
                                                                        data_set_target_task_te.y,
                                                                        data_set_target_task_te.x)
        def ser_by_minimum_distance(D_star_te, constellation):
            return      (torch.argmin(      ( constellation.y.view((1,iotUplink.modulator.order,2))
                                             -D_star_te.y.view(   (len(D_star_te), 1 ,2))
                                            ).norm(2, dim = 2),
                                            dim = 1)         !=   D_star_te.x).numpy().mean()
        txIq,txSym = iotUplink.modulator.step(iotUplink.modulator.order,0) #pattern 0,1,2,...N-1:covers all const points
        data_set_pure_constellation =   MyDataSet({'y': torch.tensor([[z.real,z.imag] for z in txIq]),
                                                   'x': txSym}                ) # y: complex to two-dim tensor
        mSerLmmse_hDemodTe[kk,ss] =     ser_by_minimum_distance (       data_set_equalized_h_te,
                                                                        data_set_pure_constellation)
        mSerLmmse_xDemodTe[kk,ss] =     ser_by_minimum_distance (       data_set_equalized_x_te,
                                                                        data_set_pure_constellation)
        ###### Benchmark: noise free (just tx non linearities) + minimum distance decoding
        data_set_min_dist_decoding =    iotUplink.step(iotUplink.modulator.order, True, True)  # noise free
        mSerOptimalDemodTe[kk, ss] =    ser_by_minimum_distance (       data_set_target_task_te,
                                                                        data_set_min_dist_decoding)
        ##### Conventional learninig using ADAM ######
        phi_conventional , vLoss_conventional =     AdaptNnUsingAdam(   xi_untrained,           # net_init
                                                                        data_set_target_task_tr,# dataset_tr
                                                                        0.1,                   # lr_first_sgd_steps
                                                                        numSgdSteps,            # numSgdSteps=1,
                                                                        16,                     # batch_size
                                                                        vConventionalAdaptIter, # vMetaAdaptIter
                                                                        0)                      # weight_decay
        for ii,iters in enumerate(vConventionalAdaptIter):
            mSerConventionalTe[kk,ss,ii] , _ =  SerOfSoftDemodNn(       phi_conventional[ii],
                                                                        data_set_target_task_te)
        ## freq ml
        def meta_adapt_from_xi(net_init, num_first_samples):
            phi =       AdaptNnUsingGd(             net_init,               # net_init
                                                    data_set_target_task_tr,# data_set_tr
                                                    lr_adapt,               # lr_first_sgd_steps
                                                    numSgdSteps,            # numSgdSteps
                                                    numMetaAdaptIter,       # numMetaAdaptIter
                                                    num_first_samples)      # num_first_samples_for_first_sgd_steps
            ser , soft_out=       SerOfSoftDemodNn( phi,
                                                    data_set_target_task_te   )
            return ser, soft_out
        ###### Meta-Adapt starting with the trained hyperparameter ######
        #mSerMamlVerATe[kk, ss],mEceMamlVerATe[kk, ss],_= meta_adapt_from_xi(  v_xi_adapt_VerA[kk],
        #                                                                    numPilotsForMetaTraining)
        mSerFMamlTe[kk, ss], fnet_soft_out =         meta_adapt_from_xi( v_xi_adapt[kk],
                                                                            numPilotsForMetaTraining)


        ## bayesian ml
        ###### Meta-Adapt starting with the trained hyperparameter ######
        varphi_star_params =            BaysianAdaptNnUsingGd(
                                                        v_bxi_adapt_params[kk],# bnet_init
                                                        data_set_target_task_tr, # data_set_tr
                                                        lr_adapt,                # lr_first_sgd_steps
                                                        numSgdSteps,             # numSgdSteps
                                                        numMetaAdaptIter,        # numMetaAdaptIter
                                                        numPilotsForMetaTraining,# num_first_samples_for_first_sgd_steps
                                                        numNets,                 # numNets
                                                        bDegenerateToFrequentist,# bDegenerateToFrequentist
                                                        'mad:' + metatest_str + nummtrtasks_str + 'bayesian: ')#print_str
        mSerBMamlTe      [kk, ss], bnet_soft_out =          BayesianMetaTest(
                                                                    varphi_star_params,
                                                                    data_set_target_task_te,
                                                                    numNets)
        if ss==0: # start new tensors
            l_dataset_x_all_test_targets_concat.append(             data_set_target_task_te.x   )
            l_fnet_soft_out_all_test_targets_concat.append(         fnet_soft_out               )
            l_bnet_soft_out_all_test_targets_concat.append(         bnet_soft_out               )
        else: # append to existing
            l_dataset_x_all_test_targets_concat[kk] =     torch.cat( (  l_dataset_x_all_test_targets_concat[kk],
                                                                        data_set_target_task_te.x ),
                                                                     dim =0 )
            l_fnet_soft_out_all_test_targets_concat[kk] = torch.cat( (  l_fnet_soft_out_all_test_targets_concat[kk],
                                                                        fnet_soft_out ),
                                                                     dim =0 )
            l_bnet_soft_out_all_test_targets_concat[kk] = torch.cat( (  l_bnet_soft_out_all_test_targets_concat[kk],
                                                                        bnet_soft_out ),
                                                                     dim =0 )
    print(print_str+metatest_str+f"time from start {datetime.datetime.now() - startTime}")
    if ((ss+1)%2==0) and (ss>0):
        for kk, numMetaTrainingTasks in enumerate(vMetaTrainingTasks):
            ## freq ml
            mEcePiledFMamlTe[kk,ss],rel_diagram_f =   EceOfSoftDemodNn(   l_fnet_soft_out_all_test_targets_concat[kk],
                                                                           l_dataset_x_all_test_targets_concat[kk])
            print(print_str+metatest_str+'Saving to file: rel_diagram_f'+str(numMetaTrainingTasks)+'.mat') # overiding
            savemat(path_of_run + 'rel_diagram_f'+str(numMetaTrainingTasks)+'.mat', {
                                                                        "rel_diagram"   : rel_diagram_f,
                                                                        "ece"           : mEcePiledFMamlTe[kk,ss],
                                                                        "ss"            : ss})
            ## bayesian ml
            mEcePiledBMamlTe[kk,ss],rel_diagram_b =   EceOfSoftDemodNn(   l_bnet_soft_out_all_test_targets_concat[kk],
                                                                           l_dataset_x_all_test_targets_concat[kk])
            print(print_str+metatest_str+'Saving to file: rel_diagram_b'+str(numMetaTrainingTasks)+'.mat') # overiding
            savemat(path_of_run + 'rel_diagram_b'+str(numMetaTrainingTasks)+'.mat', {
                                                                        "rel_diagram"   : rel_diagram_b,
                                                                        "ece"           : mEcePiledBMamlTe[kk,ss],
                                                                        "ss"            : ss})
        mSerOptimalDemodTePer  [:,:-1] =    np.percentile(  mSerOptimalDemodTe  [:,:ss],    tPercentiles, axis=1).T
        mSerLmmse_hDemodTePer  [:,:-1] =    np.percentile(  mSerLmmse_hDemodTe  [:,:ss],    tPercentiles, axis=1).T
        mSerLmmse_xDemodTePer  [:,:-1] =    np.percentile(  mSerLmmse_xDemodTe  [:,:ss],    tPercentiles, axis=1).T
        mSerFMamlTePer         [:,:-1] =    np.percentile(  mSerFMamlTe         [:,:ss],    tPercentiles, axis=1).T
        mSerBMamlTePer         [:,:-1] =    np.percentile(  mSerBMamlTe         [:,:ss],    tPercentiles, axis=1).T
        for ii,iters in enumerate(vConventionalAdaptIter):
            mSerConventionalTePer[:,:-1,ii]= np.percentile( mSerConventionalTe  [:,:ss,ii], tPercentiles, axis=1).T
        mSerOptimalDemodTePer  [:, -1] =    np.mean(        mSerOptimalDemodTe  [:,:ss],                  axis=1)
        mSerLmmse_hDemodTePer  [:, -1] =    np.mean(        mSerLmmse_hDemodTe  [:,:ss],                  axis=1)
        mSerLmmse_xDemodTePer  [:, -1] =    np.mean(        mSerLmmse_xDemodTe  [:,:ss],                  axis=1)
        mSerFMamlTePer         [:, -1] =    np.mean(        mSerFMamlTe         [:,:ss],                  axis=1)
        mSerBMamlTePer         [:, -1] =    np.mean(        mSerBMamlTe         [:,:ss],                  axis=1)
        for ii,iters in enumerate(vConventionalAdaptIter):
            mSerConventionalTePer[:, -1,ii]= np.mean(       mSerConventionalTe  [:,:ss,ii],               axis=1)

        print(print_str+metatest_str+'Saving to file: IotBayesianMaml.mat')
        print('mSerFMamlTePer')
        print(mSerFMamlTePer)
        print('mSerBMamlTePer')
        print(mSerBMamlTePer)
        savemat(path_of_run + "IotBayesianMaml.mat",
                                    {   "mSerOptimalDemodTePer"             : mSerOptimalDemodTePer,
                                        "mSerLmmse_hDemodTePer"             : mSerLmmse_hDemodTe,
                                        "mSerLmmse_xDemodTePer"             : mSerLmmse_xDemodTe,
                                        "mSerFMamlTePer"                    : mSerFMamlTePer,
                                        "mSerBMamlTePer"                    : mSerBMamlTePer,
                                        "mSerConventionalTePer"             : mSerConventionalTePer,
                                        "vConventionalAdaptIter"            : vConventionalAdaptIter,
                                        "mEcePiledFMamlTe"                  : mEcePiledFMamlTe,
                                        "mEcePiledBMamlTe"                  : mEcePiledBMamlTe,
                                        "numPilotsInFrame"                  : numPilotsInFrame,
                                        "numPilotsForMetaTraining"          : numPilotsForMetaTraining,
                                        "tPercentiles"                      : tPercentiles,
                                        "numTargetStates"                   : numTargetStates,
                                        "lr"                                : lr,
                                        "lr_shared"                         : lr_shared,
                                        "lr_adapt"                          : lr_adapt,
                                        "vMetaTrainingTasks"                : vMetaTrainingTasks,
                                        "vLayers"                           : vLayers,
                                        "TotalChannelstates"                : ss+1,
                                        "dSetting"                          : dSetting
                                    })
print('demod_run0.py done. Total time:')
print(datetime.datetime.now() - startTime)
