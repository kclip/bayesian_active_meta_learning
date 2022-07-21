import os.path

from baml4eq import *
import torch
import numpy as np
import datetime
from scipy.io import loadmat, savemat
from copy import deepcopy

# pas = PASsive Bayesian meta-learning
# fma = Fixed Mapping (phi to theta) Active Bayesian meta-learning

path_of_run =                       'run/'

startTime =                         datetime.datetime.now()

ml_ind =                            0
ml_file_str =                       path_of_run+f'ml_{ml_ind}_mtr_fma_pas.mat'

while os.path.exists(ml_file_str):
    lm =                                loadmat(ml_file_str)
    numPilotsForMetaTraining =          lm["numPilotsForMetaTraining"].item()
    numTotalMtrTasks =                  lm["numTotalMtrTasks"].item()    # as much as mtr progressed (can still be running)
    numSgdSteps =                       lm["numSgdSteps"].item()
    lr =                                lm["lr"].item()
    numMetaTrainingTasks =              lm["numMetaTrainingTasks"].item()
    numNets =                           lm["numNets"].item()
    snr_dB =                            lm["snr_dB"].item()
    modKey =                            lm["modKey"].item()
    bDegenerateToFrequentist =          lm["bDegenerateToFrequentist"].item()
    numTasksInitExperience =            lm["numTasksInitExperience"].item()

    dSetting =                          {   'snr_dB'     : snr_dB,           # Signal to noise ratio per one RX
                                            'modKey'     : modKey }
    p2pSimo =                           P2pSimo(dSetting)

    numTargetStates =                   100
    tPercentiles =                      (90, 95, 99, 10, 50, 70, 85)
    num_target_test_symbols =           1000
    numPilotsInFrame =                  numPilotsForMetaTraining
    lr_adapt =                          lr
    numMetaAdaptIter =                  int(numSgdSteps)

    numEffectiveMetaTrainingTasks =     numMetaTrainingTasks - numTasksInitExperience
    mMsePMamlTePer =                    np.zeros((numEffectiveMetaTrainingTasks, len(tPercentiles) + 1))  # P=  Passive  (Bayesian)
    mMseFMamlTePer =                    np.zeros((numEffectiveMetaTrainingTasks, len(tPercentiles) + 1))  # F=  Fixed-model Active   (Bayesian)
    mMsePMamlTe =                       np.zeros((numEffectiveMetaTrainingTasks, numTargetStates))
    mMseFMamlTe =                       np.zeros((numEffectiveMetaTrainingTasks, numTargetStates))
    mChannelStates =                    np.zeros((                      numTargetStates       , 2))
    m_loss_pas =                        np.zeros((numTargetStates, numTotalMtrTasks, numMetaAdaptIter ))
    m_loss_fma =                        np.zeros((numTargetStates, numTotalMtrTasks, numMetaAdaptIter ))
    phiMaxCandidate =                   torch.sqrt(10.0 ** (torch.tensor(snr_dB) / 10) / 4)
    print_str = f'ml {ml_ind} mda: '
    print(print_str+'Exploring MetaAdapting')
    for ss in range(numTargetStates):   # target tasks
        metatest_str =                          f"num mte tasks {ss}/{numTargetStates}: "
        p2pSimo.draw_channel_state()            # new channel state using its prior
        mChannelStates[ss,0:2] =                p2pSimo.channelState.detach().clone().numpy()
        data_set_target_task_te =               p2pSimo.step(num_target_test_symbols, False, False)  # data
        data_set_target_task_tr =               p2pSimo.step(numPilotsInFrame,        True, False)  # pilots
        for kk, mtr_task in enumerate(range(numTasksInitExperience,numMetaTrainingTasks)):
            nummtrtasks_str =               f'num mtr tasks {kk}/{numMetaTrainingTasks}: '

            bxi_pas = torch.load(f'{path_of_run}bxi_pas_ml_{ml_ind}_tasks_{mtr_task}_mtr_tasks.pt')
            bxi_fma = torch.load(f'{path_of_run}bxi_fma_ml_{ml_ind}_tasks_{mtr_task}_mtr_tasks.pt')
            with torch.no_grad():
                if bxi_pas[0].norm() > phiMaxCandidate: # makes sure the starting mean is inside the region of interest
                    bxi_pas[0] /= bxi_pas[0].norm()
                if bxi_fma[0].norm() > phiMaxCandidate:
                    bxi_fma[0] /= bxi_fma[0].norm()
                ## bayesian ml - passive
            ###### Meta-Adapt starting with the trained hyperparameter ######
            varphi_star_pas_params,\
            m_loss_pas[ss,kk,:] =               BaysianAdaptNnUsingGd(
                                                            bxi_pas,                 # bnet_init
                                                            data_set_target_task_tr, # data_set_tr
                                                            lr_adapt,                # lr_first_sgd_steps
                                                            numSgdSteps,             # numSgdSteps
                                                            numMetaAdaptIter,        # numMetaAdaptIter
                                                            numPilotsForMetaTraining,# num_first_samples_for_first_sgd_steps
                                                            numNets,                 # numNets
                                                            bDegenerateToFrequentist,# bDegenerateToFrequentist
                                                            p2pSimo,                 # p2pSimo (for modulation)
                                                            'mad:' + metatest_str + nummtrtasks_str + 'pas: ')#print_str
            mMsePMamlTe      [kk, ss] =         BayesianMetaTest( varphi_star_pas_params,
                                                                   data_set_target_task_te,
                                                                   numNets)

            #### bayesian ml - fixed-model active
            ######## Meta-Adapt starting with the trained hyperparameter ######
            varphi_star_fma_params,\
            m_loss_fma[ss,kk,:] =                BaysianAdaptNnUsingGd(
                                                            bxi_fma,                 # bnet_init
                                                            data_set_target_task_tr, # data_set_tr
                                                            lr_adapt,                # lr_first_sgd_steps
                                                            numSgdSteps,             # numSgdSteps
                                                            numMetaAdaptIter,        # numMetaAdaptIter
                                                            numPilotsForMetaTraining,# num_first_samples_for_first_sgd_steps
                                                            numNets,                 # numNets
                                                            bDegenerateToFrequentist,# bDegenerateToFrequentist
                                                            p2pSimo,                 # p2pSimo (for modulation)
                                                            'mad:' + metatest_str + nummtrtasks_str + 'fma: ')#print_str
            mMseFMamlTe      [kk, ss] =          BayesianMetaTest( varphi_star_fma_params,
                                                                   data_set_target_task_te,
                                                                   numNets)

        print(print_str+metatest_str+f"time from start {datetime.datetime.now() - startTime}")
        if (((numTargetStates<=6) or (ss+1)%5==0) and (ss>0)):
            mMsePMamlTePer         [:,:-1] =    np.percentile(  mMsePMamlTe         [:,:ss],    tPercentiles, axis=1).T
            mMseFMamlTePer         [:,:-1] =    np.percentile(  mMseFMamlTe         [:,:ss],    tPercentiles, axis=1).T
            mMsePMamlTePer         [:, -1] =    np.mean(        mMsePMamlTe         [:,:ss],                  axis=1)
            mMseFMamlTePer         [:, -1] =    np.mean(        mMseFMamlTe         [:,:ss],                  axis=1)
            print(print_str+metatest_str+'Saving to file: meta_testing.mat')
            print('mMsePMamlTePer')
            print(mMsePMamlTePer)
            print('mMseFMamlTePer')
            print(mMseFMamlTePer)
            file_str =                      f'ml_{ml_ind}_mte_losses'
            print(f'saving after {ss} meta testing channels states to {file_str}.mat')
            savemat(f'{path_of_run}{file_str}.mat',
                                        {   "mMsePMamlTePer"                    : mMsePMamlTePer,
                                            "mMseFMamlTePer"                    : mMseFMamlTePer,
                                            "numTargetStates"                   : numTargetStates,
                                            "tPercentiles"                      : tPercentiles,
                                            "num_target_test_symbols"           : num_target_test_symbols,
                                            "numPilotsInFrame"                  : numPilotsInFrame,
                                            "lr_adapt"                          : lr_adapt,
                                            "numMetaAdaptIter"                  : numMetaAdaptIter,
                                            "vMetaTrainingTasks"                : np.arange(numTasksInitExperience,numMetaTrainingTasks),
                                            "TotalChannelstates"                : ss+1,
                                            "mChannelStates"                    : mChannelStates,
                                            "m_loss_pas"                        : m_loss_pas,
                                            "m_loss_fma"                        : m_loss_fma,
                                        })

    ml_ind +=               1 # try the next met-learning round
    ml_file_str =           path_of_run + f'ml_{ml_ind}_mtr_fma_pas.mat'

print(f'Done running after {datetime.datetime.now() - startTime}')
