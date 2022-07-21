from baml4eq import *
import torch
import numpy as np
import datetime
from scipy.io import loadmat, savemat
from copy import deepcopy
import os

# pas = PASsive Bayesian meta-learning
# fma = Fixed Mapping (phi to theta) Active Bayesian meta-learning

path_of_run =                       'run/'

startTime =                         datetime.datetime.now()

numMetaLearnings =                  100
numMetaTrainingTasks =              14
numPilotsForMetaTraining =          4
numDataSymForMetaTraining =         4
numSymbolsInMetaTrainingTask =      numPilotsForMetaTraining + numDataSymForMetaTraining
numMetaTrainingInit =               100
numMetaTrainingMin =                100
numNets =                           100
numSgdSteps =                       2
numTasksInitExperience =            3
tasks_batch_size_mtr =              100
lr =                                2e-3
lr_shared =                         5e-2
lr_phi =                            5e-2
numPhiGaSteps =                     5
bCreateGraphForXi =                 (numSgdSteps < 100) and (numMetaTrainingInit>1) # faster run when False - for inspecting only the adaptation phase
snr_dB =                            6.0
dSetting =                          {   'snr_dB'     : snr_dB,           # Signal to noise ratio per one RX
                                        'modKey'     : '4PAM' }
bFirstOrderApprox =                 False # True=FOMAML False=MAML
p2pSimo_pas =                       P2pSimo(dSetting)
p2pSimo_fma =                       P2pSimo(dSetting)
active_roi_phi_str =                'enclosed_circle' # 'square','enclosed_circle';
numPhiPerAxis =                     201 # 401 is fine for high resolution, 101 for low
phiMaxCandidate =                   torch.sqrt(10.0 ** (torch.tensor(snr_dB) / 10) / 4) # highest possible value of abs(phi)
l_ml_ind_nan_spotted =              []


if not os.path.exists(path_of_run):
    os.makedirs(path_of_run)
    print(f'Directory {path_of_run} not found, making it.')

for ml_ind in range(numMetaLearnings): # index of meta_learning
    # new learning, everything is new
    l_mtr_dataset_pas =             []
    l_mtr_dataset_fma =             []
    l_mtr_channel_states_pas =      []
    l_mtr_channel_states_fma =      []
    footprint_next_phi_pas =        np.zeros([numMetaTrainingTasks , 2])  # extra one for the init phi
    footprint_next_phi_fma =        np.zeros([numMetaTrainingTasks , 2])  # extra one for the init phi
    footprint_mChannelStates_pas =  np.zeros((numMetaTrainingTasks , 2))
    footprint_mChannelStates_fma =  np.zeros((numMetaTrainingTasks , 2))

    for kk,mtr_task in enumerate(range(0,numTasksInitExperience)): # same init mtr tasks and data sets for all schemes
        p2pSimo_pas.draw_channel_state()
        channel_state =                                 p2pSimo_pas.channelState
        phi_mmse =                                      phi_to_channelstate_by_min_norm(channel_state)
        mtr_data_set =                                  p2pSimo_pas.step(numSymbolsInMetaTrainingTask, True, False)
        l_mtr_dataset_pas.append(                       mtr_data_set)
        l_mtr_dataset_fma.append(                       deepcopy(mtr_data_set))
        l_mtr_channel_states_pas.append(                channel_state.detach().clone())  # the same channel state
        l_mtr_channel_states_fma.append(                channel_state.detach().clone())  # the same channel state
        footprint_next_phi_pas      [mtr_task, 0:2] =   phi_mmse.numpy()
        footprint_next_phi_fma      [mtr_task, 0:2] =   phi_mmse.numpy()
        footprint_mChannelStates_pas[mtr_task, 0:2] =   channel_state.detach().clone().numpy()
        footprint_mChannelStates_fma[mtr_task, 0:2] =   channel_state.detach().clone().numpy()

    nu_pas_init =                       MyLinear(2)
    varrho_pas_init =                   MyLinear(2)
    for w in nu_pas_init.parameters():
        w.data = torch.tensor([-0.2,-0.2]) # starting on the top point of the enclosed circle
    for w in varrho_pas_init.parameters():
        w.data = torch.tensor([-1.0, -1.5]) ##########################  torch.tensor([-1.5, -2.5])
    bDegenerateToFrequentist =          False
    for w in varrho_pas_init.parameters():
        if bDegenerateToFrequentist:  # we degenerate bayesian to freq by learning only the mean nu and
            w.requires_grad_(False) # disables them as parameters while creating a graph...
            w -= np.inf   # set all varrho parameters to -inf, which means std=exp(-inf)=0 (not stochastic)
    bxi_pas_init =                      (   list(map(lambda p: p[0], zip(nu_pas_init.parameters()      )))
                                          + list(map(lambda p: p[0], zip(varrho_pas_init.parameters()  )))    )
    bxi_fma_init =                      list(map(lambda p: deepcopy(p[0]), zip(bxi_pas_init)))

    vPhi0 =                         torch.linspace(-phiMaxCandidate, phiMaxCandidate, steps=numPhiPerAxis)
    vPhi1 =                         torch.linspace(-phiMaxCandidate, phiMaxCandidate, steps=numPhiPerAxis)
    mPhi0, mPhi1 =                  torch.meshgrid(vPhi0, vPhi1)
    vPhi0_flatten =                 mPhi0.flatten()
    vPhi1_flatten =                 mPhi1.flatten()
    if active_roi_phi_str=='square':
        vPhi_valid =                torch.ones_like(vPhi0_flatten)
    elif active_roi_phi_str=='enclosed_circle':
        vPhi_valid =                1.0*(vPhi0_flatten**2 + vPhi1_flatten**2 <= phiMaxCandidate**2)
    vScoreOfPhi =                       torch.zeros_like(vPhi0_flatten)

    v_bxi_pas =                         []
    v_bxi_fma =                         []
    v_dMtr_losses_fma =                 []
    v_dMtr_losses_pas =                 []
    v_footprint_fma =                   []
    v_footprint_pas =                   []
    v_numMetaTraining =                 np.zeros(numMetaTrainingTasks,dtype='int')

    footprint_vScoreOfPhiOverIters_fma =    np.zeros([numMetaTrainingTasks-numTasksInitExperience,len(vScoreOfPhi)  ])
    footprint_vScoreOfPhiOverIters_pas =    np.zeros([numMetaTrainingTasks-numTasksInitExperience,len(vScoreOfPhi)  ])
    footprint_phi_while_ga_act =            np.zeros((numMetaTrainingTasks-numTasksInitExperience,numPhiGaSteps+1, 2))
    footprint_scoring_while_ga_act =        np.zeros((numMetaTrainingTasks-numTasksInitExperience,numPhiGaSteps+1   ))
    footprint_phi_ga_epochs_act =           np.zeros((numMetaTrainingTasks-numTasksInitExperience,                  ))


    print_str =                         f'ml {ml_ind}/{numMetaLearnings} '
    for kk, mtr_task in enumerate(range(numTasksInitExperience,numMetaTrainingTasks)):
        active_iter_str = f'task {mtr_task+1}/{numMetaTrainingTasks} '
        print(print_str + f" MetaTraining with {mtr_task+1} meta-training tasks")
        # Mind this crucial step! instead of starting from the last learnt xi, we start from the same one for each round
        if True or mtr_task==numTasksInitExperience: # start freshly on the first i
            bnet_params_init_fma =          list(map(lambda p: deepcopy(p[0]), zip(bxi_fma_init)))
            bnet_params_init_pas =          list(map(lambda p: deepcopy(p[0]), zip(bxi_pas_init)))
        else: # continue from the last adopted till now
            bnet_params_init_fma =          list(map(lambda p: deepcopy(p[0]), zip(bxi_fma)))
            bnet_params_init_pas =          list(map(lambda p: deepcopy(p[0]), zip(bxi_pas)))
        # many iterations for the start of the incremental learning to give chance for few tasks.
        # gradually use less iterations, as most of the learning was done and only small updates are needed
        # maintain a minimum of numMetaTrainingMin iters
        v_numMetaTraining[kk] =             ( max(np.int(numMetaTrainingInit/(kk+1)) , np.int(numMetaTrainingMin) ))

        bxi_pas,\
        _,\
        dMtr_losses_pas, \
        v_varphi_pas_adapt,\
        footprint_pas =             BayesianMamlMetaTrain(
                                                    bnet_params_init_pas,           # bnet_params_init
                                                    l_mtr_dataset_pas,              # data_set_meta_train
                                                    lr,                             # lr=0.1
                                                    lr_shared,                      # lr_shared=0.1,
                                                    tasks_batch_size_mtr,           # tasks_batch_size_mtr=20,
                                                    numSgdSteps,                    # numSgdSteps=1,
                                                    numPilotsForMetaTraining,       # numPilotsForMetaTraining=16,
                                                    v_numMetaTraining[kk],          # numMetaTrainingIter=10000
                                                    numNets,                        # numNets
                                                    bFirstOrderApprox,              # bFirstOrderApprox ,TRUE=FOMAML
                                                    bDegenerateToFrequentist,       # bDegenerateToFrequentist
                                                    l_mtr_channel_states_pas,       # l_mtr_channel_states, # needed for MRC or MMSE testbenches
                                                    snr_dB,                         # snr_dB
                                                    print_str + active_iter_str + ' pas ')           # print_str
        v_bxi_pas.append(               bxi_pas)
        v_dMtr_losses_pas.append(       dMtr_losses_pas)
        v_footprint_pas.append(         footprint_pas)
        torch.save(bxi_pas, f'{path_of_run}bxi_pas_ml_{ml_ind}_tasks_{mtr_task}_mtr_tasks.pt')

        bxi_fma,\
        _,\
        dMtr_losses_fma, \
        v_varphi_fma_adapt,\
        footprint_fma=           BayesianMamlMetaTrain(
                                                    bnet_params_init_fma,           # bnet_params_init
                                                    l_mtr_dataset_fma,              # data_set_meta_train
                                                    lr,                             # lr=0.1
                                                    lr_shared,                      # lr_shared=0.1,
                                                    tasks_batch_size_mtr,           # tasks_batch_size_mtr=20,
                                                    numSgdSteps,                    # numSgdSteps=1,
                                                    numPilotsForMetaTraining,       # numPilotsForMetaTraining=16,
                                                    v_numMetaTraining[kk],          # numMetaTrainingIter=10000
                                                    numNets,                        # numNets
                                                    bFirstOrderApprox,              # bFirstOrderApprox ,TRUE=FOMAML
                                                    bDegenerateToFrequentist,       # bDegenerateToFrequentist
                                                    l_mtr_channel_states_fma,       # l_mtr_channel_states, # needed for MRC or MMSE testbenches
                                                    snr_dB,                         # snr_dB
                                                    print_str + active_iter_str + 'fma ')           # print_str
        v_bxi_fma.append(               bxi_fma)
        v_dMtr_losses_fma.append(       dMtr_losses_fma)
        v_footprint_fma.append(footprint_fma)
        torch.save(bxi_fma,  f'{path_of_run}bxi_fma_ml_{ml_ind}_tasks_{mtr_task}_mtr_tasks.pt')

        ## pas scoring of latent space
        vScoreOfPhi =           torch.zeros_like(mPhi0).flatten()
        best_scoring_so_far =   torch.tensor(-float('inf'))
        for i, (phi0, phi1, phi_valid) in enumerate(zip(vPhi0_flatten,vPhi1_flatten,vPhi_valid)):
            candidate_phi =     torch.stack((phi0, phi1))
            vScoreOfPhi[i] =    scoring_function_of_candidate(candidate_phi, v_varphi_pas_adapt)
            if (vScoreOfPhi[i]>best_scoring_so_far) and (phi_valid==1): # explore only those in Region of interest
                best_scoring_so_far =               vScoreOfPhi[i].detach().clone()
                best_phi_pas =                      candidate_phi.detach().clone()
        footprint_vScoreOfPhiOverIters_pas[kk,:] =      vScoreOfPhi.numpy()
        # pas next
        p2pSimo_pas.draw_channel_state()
        next_channnel_state_pas =               p2pSimo_pas.channelState         # keep track on the drawn one
        footprint_next_phi_pas[mtr_task, 0:2] = phi_to_channelstate_by_min_norm(   next_channnel_state_pas)
        footprint_mChannelStates_pas[mtr_task,0:2] = p2pSimo_pas.channelState.detach().clone().numpy()
        print('mChannelStates_pas')
        print(footprint_mChannelStates_pas)
        l_mtr_dataset_pas.append(               p2pSimo_pas.step(numSymbolsInMetaTrainingTask, True, False)) # append next task's data set
        l_mtr_channel_states_pas.append(        p2pSimo_pas.channelState)  # append next task

        ## fma scoring of latent space
        vScoreOfPhi =           torch.zeros_like(mPhi0).flatten()
        best_scoring_so_far =   torch.tensor(-float('inf'))
        for i, (phi0, phi1, phi_valid) in enumerate(zip(vPhi0_flatten,vPhi1_flatten,vPhi_valid)):
            candidate_phi =     torch.stack((phi0, phi1))
            vScoreOfPhi[i] =    scoring_function_of_candidate(candidate_phi, v_varphi_fma_adapt)
            if (vScoreOfPhi[i]>best_scoring_so_far) and (phi_valid==1): # explore only those in Region of interest
                best_scoring_so_far =               vScoreOfPhi[i].detach().clone()
                best_phi_fma =                      candidate_phi.detach().clone()
        footprint_vScoreOfPhiOverIters_fma[kk,:] =      vScoreOfPhi.numpy()
        # fma next
        next_channnel_state_fma =                       phi_to_channelstate_by_min_norm(best_phi_fma)
        p2pSimo_fma.set_channel_state_from_vec(         next_channnel_state_fma)
        footprint_next_phi_fma[mtr_task, 0:2] =         best_phi_fma.numpy()
        footprint_mChannelStates_fma[mtr_task,0:2] =    p2pSimo_fma.channelState.detach().clone().numpy()
        print('mChannelStates_fma')
        print(footprint_mChannelStates_fma)

        l_mtr_dataset_fma.append(               p2pSimo_fma.step(numSymbolsInMetaTrainingTask, True, False)) # append next task's data set
        l_mtr_channel_states_fma.append(        p2pSimo_fma.channelState)  # append next task

        file_str =                              f'ml_{ml_ind}_mtr_fma_pas.mat'
        savemat(path_of_run+file_str,       {   "vScoreOfPhiOverIters_pas":     footprint_vScoreOfPhiOverIters_pas,
                                                "vScoreOfPhiOverIters_fma":     footprint_vScoreOfPhiOverIters_fma,
                                                "v_varphi_pas":                 v_footprint_pas,
                                                "v_varphi_fma":                 v_footprint_fma,
                                                "numSymbolsInMetaTrainingTask": numSymbolsInMetaTrainingTask,
                                                "numPilotsForMetaTraining":     numPilotsForMetaTraining,
                                                "numTotalMtrTasks":             kk+1,
                                                "numSgdSteps":                  np.array(numSgdSteps,dtype=np.single),
                                                "beta_mse":                     np.array(BETA_MSE,dtype=np.single),
                                                "v_dMtr_losses_pas":            v_dMtr_losses_pas,
                                                "v_dMtr_losses_fma":            v_dMtr_losses_fma,
                                                "numPhiPerAxis":                numPhiPerAxis,
                                                "phiMaxCandidate":              phiMaxCandidate.numpy(),
                                                "vPhi_valid":                   vPhi_valid.numpy(),
                                                "next_phi_pas":                 footprint_next_phi_pas,
                                                "next_phi_fma":                 footprint_next_phi_fma,
                                                "lr":                           lr,
                                                "numMetaTrainingTasks":         numMetaTrainingTasks,
                                                "numNets":                      numNets,
                                                "snr_dB":                       snr_dB,
                                                "modKey":                       dSetting['modKey'],
                                                "bDegenerateToFrequentist":     bDegenerateToFrequentist,
                                                "v_numMetaTraining":            v_numMetaTraining,
                                                "mChannelStates_pas":           footprint_mChannelStates_pas,
                                                "mChannelStates_fma":           footprint_mChannelStates_fma,
                                                "phi_while_ga_act":             footprint_phi_while_ga_act,
                                                "scoring_while_ga_act":         footprint_scoring_while_ga_act,
                                                "phi_ga_epochs_act":            footprint_phi_ga_epochs_act,
                                                "numTasksInitExperience":       numTasksInitExperience,
                                                "l_ml_ind_nan_spotted":         l_ml_ind_nan_spotted

        } )   # override to the latest mtr tasks
        print('Saving to file '+file_str)
print(f'Done running mtr after {datetime.datetime.now() - startTime}')


import runpy
runpy.run_path(path_name='main_toy_mte.py')


print(f'Done running mtr+mte after {datetime.datetime.now() - startTime}')