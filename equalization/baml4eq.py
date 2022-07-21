import torch
from torch import nn
import numpy as np
import datetime
from copy import deepcopy

REPARAM_COEFF =     1.0
KL_TEMP =           1.0
BETA_MSE =          150 # 60 # precision for the Gaussian assumption of the linear combiner p(x|y,phi)=N(phi^T*y,beta^-1)
ALPHA_RECONST =     1.0
PRINT_INTERVAL =    20  # print losses each this number of iterations

##################

class Modulation:
    def __init__(self, modKey):
        self.modKey = modKey
        dOrders =       {   'WGN':  [],
                            '4PAM': 4}
        dNormCoeffs =   {   'WGN':  torch.tensor(1.0),
                            '4PAM': torch.tensor(1 /  5.0).sqrt()}
        dMappings =     {   'WGN':  [],
                            '4PAM': torch.tensor( [-3.0,-1.0,+1.0,+3.0])}
        self.order = dOrders[modKey]
        self.normCoeff = dNormCoeffs[modKey]
        self.vMapping = dMappings[modKey]

    def modulate(self,vSamplesUint):
        if self.modKey == 'WGN':
            return vSamplesUint
        else:
            return self.normCoeff * self.vMapping[vSamplesUint]

    def step(self, numSamples, xPattern ):
        if self.modKey == 'WGN':
            vSamplesUint =              torch.randn(numSamples) # not really uint, just for generalization
            vSamplesReal =              vSamplesUint
            return (vSamplesReal, vSamplesUint)
        else:
            if xPattern==0:             # 0,1,2,3,0,1,2,3,0,...
                vSamplesUint =          torch.arange(0,numSamples, dtype = torch.long)   %self.order
            else:  # some random balanced permutation of (0,1,...M-1,0,1,...M-1,...)
                vSamplesUint =          torch.randperm(numSamples) % self.order
            vSamplesReal =              self.modulate(vSamplesUint)
            return (vSamplesReal, vSamplesUint)

class MyDataSet(torch.utils.data.Dataset):

    # Constructor
    def __init__ ( self, dDataset):   # dDataset['y'] are the inputs and dDataset['x'] (long) are the outputs on purpose
        self.y =        dDataset['y']
        self.x =        dDataset['x']
        self.len =      len(self.x)

    # Getter
    def __getitem__ ( self, index ):
        return self.y[index], self.x[index]

    # Get length
    def __len__ ( self ):
        return self.len

    def split_into_two_subsets( self, numSamFirstSubSet, bShuffle = True ): # including shuffling
        N =             self.len
        N0 =            numSamFirstSubSet
        N1 =            N - N0
        if N0==N:
            return (    MyDataSet({     'y': self.y,
                                        'x': self.x  } ),
                        MyDataSet({     'y': [],
                                        'x': []  } )       )
        elif N1==N:
            return (    MyDataSet({     'y': [],
                                        'x': []}),
                        MyDataSet({     'y': self.y,
                                        'x': self.x }  )   )
        else:
            if bShuffle:
                perm =  torch.randperm(N)
            else:
                perm =  torch.arange(0,N)
            return (    MyDataSet({     'y': self.y[perm[    : N0],:],
                                        'x': self.x[perm[    : N0]  ]  } ),
                        MyDataSet({     'y': self.y[perm[-N1 :   ],:],
                                        'x': self.x[perm[-N1 :   ]  ]  } ) )

class P2pSimo:
    def __init__(self, dSetting):
        self.dSetting =                 dSetting
        self.modulator =                Modulation(dSetting['modKey'])
        self.snr_lin =                  10.0 ** (self.dSetting['snr_dB'] / 10)
        self.draw_channel_state()

    def draw_channel_state(self): # samples new state to be used later
        self.channelState = torch.randn(2)

    def set_channel_state_from_vec(self, channelState): # samples new state to be used later
        self.channelState =         channelState.detach().clone()

    def step(self, numSamples, bEnforcePattern, bNoiseFree = False):
        if bEnforcePattern:
            pattern =       0
        else:
            pattern =       -1
        txReal , txSym =      self.modulator.step(numSamples, pattern)
        if bNoiseFree:
            noise =         0
        else:
            noise  =        np.sqrt(1.0 /self.snr_lin)*  torch.randn(numSamples,2)
        rxReal =            self.channelState * txReal.view([-1,1]) + noise
        return MyDataSet(   {       'y': rxReal,
                                    'x': txReal  } )   # the modulated signal, not the uint!

class MyLinear(nn.Module):     # Fully-Connected ReLU Deep Neural Network
    # Constructor
    def __init__ ( self, input_size ):
        super(MyLinear, self).__init__()
        self.lin =              torch.nn.Linear(input_size, 1, bias = False)

    # Prediction
    def forward ( self, activation ):
        activation =             self.lin(activation)
        return activation


class Linear_Meta(nn.Module):
    # Constructor
    def __init__ ( self ):
        super(Linear_Meta, self).__init__() # no need to initialize nn since we are given with parameter list
    # Prediction
    def forward ( self, net_in , net_params):
        net_in =                torch.nn.functional.linear(net_in, net_params, bias = None)
        return net_in

class BayesianLinear_Meta(nn.Module):
    # Constructor
    def __init__ ( self ):
        super(BayesianLinear_Meta, self).__init__() # no need to init since we are given with parameter list
    # Prediction
    def forward ( self, bnet_in , bnet_params): # net_in is the input and output. net_params is the nn parameter vector
        weight_nu =             bnet_params[0] # in-place assignment, same id() for LHS and RHS
        weight_varrho =         bnet_params[1] # "
        weight_random =         weight_nu + torch.exp(weight_varrho) * torch.randn_like(weight_varrho)*REPARAM_COEFF
        bnet_in =               torch.nn.functional.linear(bnet_in, weight_random, bias = None)
        return bnet_in


class NnBiasedScaledSigmoids(nn.Module):    # Fully-Connected ReLU Deep Neural Network,
    # last layer is a vec of biased and scaled (fixed values) sigmoids
    def __init__ ( self, vLayers , dLimits ):
        super(NnBiasedScaledSigmoids, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(vLayers, vLayers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
        self.dLimits =      dLimits
        self.lowest =       torch.tensor(dLimits[:,0]) # shape = [M]
        self.highest =      torch.tensor(dLimits[:,1]) # shape = [M]

    # Prediction
    def forward ( self, activation ):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.nn.functional.relu(linear_transform(activation))
            else:
                activation = (self.highest-self.lowest) * torch.sigmoid(linear_transform(activation)) + self.lowest
        return activation

def DrawOnePhiByVarphi(varphi,has_bias): # if phi has weight+bais, should be 1, if only weight, should be 0
    L = int(len(varphi) / (1+has_bias) / 2) # first 2 by weight+bias, second 2 for nu+varrho L stands for number of layers
    phi = []
    for ll in range(L): ## 2*ll is the weight parameter, 2*ll+1 is the bias
        weight_nu =             varphi[               2*ll    ] # in-place assignment, same id() for LHS and RHS
        weight_varrho =         varphi[2*L*has_bias + 2*ll    ] # "
        weight_random =         weight_nu + torch.exp(weight_varrho) * torch.randn_like(weight_varrho)*REPARAM_COEFF
        phi.append(             weight_random)
        if has_bias:
            bias_nu =           varphi[               2*ll + 1]
            bias_varrho =       varphi[2*L*has_bias + 2*ll + 1]
            bias_random =       bias_nu   + torch.exp(  bias_varrho) * torch.randn_like(  bias_varrho)*REPARAM_COEFF
            phi.append(         bias_random)
    return torch.cat([w.flatten() for w in phi]) #  turn into a vector

def weighted_mse_loss(input, target, dLimits):
    weight = torch.tensor(1.0/(dLimits[:,1] - dLimits[:,0])**2)
    return (weight * (input - target) ** 2).sum()

def KL_GaussianRandomNetworks( varphi_tau_params , xi_params):
    # returns KL( q(phi_tau | varphi_tau) || p(phi_tau | xi_tau) )
    # where xi and varphi_tau each represented by {nu=mean, varrho=exp(std)}
    assert(len(varphi_tau_params)==len(xi_params))
    Q =                             int(len(xi_params)/2)
    nu_tau_params =                 varphi_tau_params    [  : Q] # in-place view
    varrho_tau_params =             varphi_tau_params    [Q :  ] # in-place view
    nu_params =                     xi_params            [  : Q] # in-place view
    varrho_params =                 xi_params            [Q :  ] # in-place view
    res =                           torch.tensor(0.0)
    for nu_tau, varrho_tau, nu, varrho in zip(nu_tau_params, varrho_tau_params, nu_params, varrho_params):
        if torch.any(torch.isinf(varrho.view(-1))) or torch.any(torch.isinf(varrho.view(-1))):
            print('warning at function KL_GaussianRandomNetworks: inf value for varrho, this could cause numerical instabilities in the KL')
        res +=      0.5 * (torch.sum(2 * varrho.view(-1) - 2 * varrho_tau.view(-1)
                                      + (  torch.exp( 2 * varrho_tau.view(-1)) + (nu_tau.view(-1) - nu.view(-1)) ** 2)
                                         * torch.exp(-2 * varrho.view(-1)   )   )
                           - nu_tau.numel()   )  # contribution of current params to KL(q||p). view(-1) vectorizes
    return res

def scoring_function_of_candidate(candidate_phi , v_varphi):
    # candidate_phi= list of 2*L, where 2=(weight+bias) and L is the number of layers in the phi NN
    # v_varphi = list of 4*L, where 4=(nu_varrho)*(weight+bias) and the same L
    v_log_scoring_tau =                     torch.zeros(len(v_varphi))
    for tt, varphi_tau in enumerate(v_varphi):
        numParams =                 int(len(varphi_tau) / 2)
        for pp in range(numParams):
            nu_tau_pp =                     varphi_tau[0        +pp] # mean     nu     is the first half values
            varrho_tau_pp =                 varphi_tau[numParams+pp] # log(std) varrho is the last  half values
            # norm pdf's
            v_log_scoring_tau[tt] +=        torch.sum(  -varrho_tau_pp
                                                        -(candidate_phi-nu_tau_pp)**2/(2*torch.exp(2*varrho_tau_pp))
                                                        -0.5*np.log(2*np.pi))# sum over layer, per mtr task tau
    bias_for_dr =                           torch.max(v_log_scoring_tau) - 0    # dynamic range of log(double_precision)
                                    # is roughly [-95,+88] to log(sum(exp())). we aim for safe upper and lower guards
    return                      - ( torch.log( torch.mean(torch.exp(v_log_scoring_tau - bias_for_dr)) ) + bias_for_dr )


def KeepOnlyRequiredGrad (params_list): #returns a sub-list keeping in-place params with attribute requires_grad = True
                                        # The parameters are a view and not a copy
    new_list = list()
    for w in params_list:
        if w.requires_grad:
            new_list.append(w)
    return new_list
    # should be the same as:     return list(filter(lambda w: w.requires_grad, bxi_untrained_params))


def KeepOnlyRequiredGradNot (params_list):  # complementary list wrt KeepOnlyRequiredGrad()
                                            # The parameters are a view and not a copy
    new_list = list()
    for w in params_list:
        if not w.requires_grad:
            new_list.append(w)
    return new_list
    # should be the same as:     return list(filter(lambda w: not w.requires_grad, bxi_untrained_params))

def channelstate_to_phi_mmse(channel_state, snr_dB):
    return channel_state / ((channel_state**2).sum()+ (10.0 ** (-snr_dB / 10)))

def phi_mmse_to_channelstate(phi_mmse, snr_dB):
    # the + before the sqrt is true only for phi with high enough energy, we assume this for high enough SNRs.
    # for lower SNRs, use - instead
    return (1+torch.sqrt(1-4*(phi_mmse**2).sum()*(10.0 ** (-snr_dB / 10))))/(2*(phi_mmse**2).sum()) * phi_mmse

def phi_to_channelstate_by_min_norm(phi_mmse):
    # the + before the sqrt is true only for phi with high enough energy, we assume this for high enough SNRs.
    # for lower SNRs, use - instead
    return phi_mmse / phi_mmse.norm()**2


def BayesianMamlMetaTrain(      bnet_params_init,                  # Bayesian net of init as list of parameters (mean+exp(std))
                                data_set_meta_train,
                                lr =                        0.1,
                                lr_shared =                 0.1,
                                tasks_batch_size_mtr =      20,
                                numSgdSteps =               1,
                                numPilotsForMetaTraining =  4,
                                numMetaTrainingIter =       10000,
                                numNets =                   100,
                                bFirstOrderApprox =         False,  # TRUE = FOMAML; FALSE = MAML with Hessians
                                bDegenerateToFrequentist =  False,  # TRUE = Frequentist ; FALSE = Bayesian
                                l_mtr_channel_states =      [],     # needed for MRC or MMSE testbenches
                                snr_dB =                    0.0,
                                print_str =                 '',
                                ):
    # bxi_params_list should be by      (   list(map(lambda p: p[0], zip(nu.parameters()    )))
    #                                     + list(map(lambda p: p[0], zip(varrho.parameters())))      )
    startTime =                         datetime.datetime.now()
    minimal_training_loss_adapt =       +np.inf
    bxi_params_adapt =                  deepcopy(bnet_params_init)
    bxi_params =                        deepcopy(bnet_params_init)
    criterion =                         torch.nn.MSELoss()
    bCreateGraphForXi =                 (numMetaTrainingIter>1) and not bFirstOrderApprox
    numMetaTrainingTasks =              len(data_set_meta_train)
    footprint_nu_tau_i =                np.zeros([numMetaTrainingIter,numMetaTrainingTasks,1+numSgdSteps,2])
    footprint_varrho_tau_i =            np.zeros([numMetaTrainingIter,numMetaTrainingTasks,1+numSgdSteps,2])
    footprint_task_loss =               np.zeros([numMetaTrainingIter,numMetaTrainingTasks,1+numSgdSteps  ])
    footprint_phi_LS =                  np.zeros([                    numMetaTrainingTasks,              2])
    footprint_phi_MRC =                 np.zeros([                    numMetaTrainingTasks,              2])
    footprint_phi_MMSE =                np.zeros([                    numMetaTrainingTasks,              2])
    footprint_Sigma_MAP_EBL =           np.zeros([                    numMetaTrainingTasks,              2,2]) # Covariance using Exact Bayesian Learning
    footprint_nu_MAP_EBL =              np.zeros([                    numMetaTrainingTasks,              2])   # mean       using Exact Bayct Bayesian Learning
    footprint_x_D_tau_tr =              np.zeros([                    numMetaTrainingTasks,numPilotsForMetaTraining   ])   # mean       using Exact Bayesian Learning
    footprint_y_D_tau_tr =              np.zeros([                    numMetaTrainingTasks,numPilotsForMetaTraining, 2])   # mean       using Exa
    optimizer_meta =                    torch.optim.Adam ( bxi_params, lr = lr_shared , betas=(0.5, 0.9)) # low momentum since we don't use minibatch
    v_varphi_adapt =                    []
    vMetaTrainingLossAdapt =            []
    vMetaTrainingAdaptWithUpdates =     []
    net_meta_intermediate =             BayesianLinear_Meta() # for single data point
    # DO NOT PLACE torch.manual_seed(0)  here  for reproducibility with other MAML schemes! this makes the channel selection for passive drawing the same values
    for meta_training_iter in range(numMetaTrainingIter):
        outer_iter_str =                f"iter {meta_training_iter+1}/{numMetaTrainingIter} "
        tasks_batch =                   np.arange(len(data_set_meta_train))[:tasks_batch_size_mtr]# consider all tasks in 0,1,... kk  #torch.randperm(len(data_set_meta_train))[:tasks_batch_size_mtr].numpy() #draw tasks minibatch
        lD_te =                         []
        lengths_lD_tr =                 torch.zeros(len(tasks_batch))
        lengths_lD_te =                 torch.zeros(len(tasks_batch))
        vLoss_te =                      torch.zeros(len(tasks_batch))
        if (meta_training_iter % PRINT_INTERVAL == 0):
            print('')
            print(f'{print_str} {outer_iter_str} nu ={bxi_params[0].detach().numpy()} exp(varrho )={torch.exp(bxi_params[1]).detach().numpy()}')
        for w in KeepOnlyRequiredGrad(bxi_params):
            w.total_grad = torch.zeros_like(w)
        for task_ind,tau in enumerate(tasks_batch):
            inner_iter_str =            f"task_ind={task_ind+1}/{len(tasks_batch)}"
            [ D_tau_tr , D_tau_te ] =   data_set_meta_train[tau].split_into_two_subsets(numPilotsForMetaTraining, False)
            lengths_lD_tr[task_ind] =   len(D_tau_tr)
            lengths_lD_te[task_ind] =   len(D_tau_te)
            lD_te.append(               D_tau_te)
            query_mini_batch_len =      len(D_tau_te) # for meta updates
            # local update, stemming from bxi_params
            bvarphi_tau_i_params =      bxi_params # initialization, to be reassigned on later updates.
            # no need to zero the grad of the LHS since the RHS comes with grad=0 and the assignment is a copied version
            if meta_training_iter==0: # calc only once, meta iter 0 was chosen arbitrary
                footprint_phi_LS[task_ind,0:2] =     (torch.inverse( D_tau_tr.y.t() @ D_tau_tr.y
                                                                  )  @ (D_tau_tr.y.t() @ D_tau_tr.x)).numpy().copy()
                footprint_phi_MMSE[task_ind,0:2] =   channelstate_to_phi_mmse(  l_mtr_channel_states[tau],
                                                                                snr_dB   ).numpy().copy()
                footprint_phi_MRC [task_ind, 0:2] =  (l_mtr_channel_states[tau] /   (l_mtr_channel_states[tau] **2
                                                                                    ).sum()).numpy().copy()
            if meta_training_iter==numMetaTrainingIter-1: # last iter
                # EBL (= Exact Bayesian Learning) using Gaussian-Gaussian GLM (=Generalized Linear Model)
                footprint_Sigma_MAP_EBL[task_ind, 0:2,0:2] = torch.inverse(  torch.diagflat(torch.exp(-2*bxi_params[1])) # varrho=varphi[1]
                                                                   + BETA_MSE * D_tau_tr.y.t() @ D_tau_tr.y ).detach().numpy()
                footprint_nu_MAP_EBL[task_ind, 0:2] =        footprint_Sigma_MAP_EBL[task_ind,0:2,0:2] @ (
                            torch.diagflat(torch.exp(-2*bxi_params[1])) @ bxi_params[0] # nu=varphi[0], varrho=varphi[1]
                        + BETA_MSE * D_tau_tr.y.t() @ D_tau_tr.x ).detach().numpy()
            footprint_nu_tau_i    [meta_training_iter,task_ind,0,0:2] = bvarphi_tau_i_params[0].detach().numpy().copy()
            footprint_varrho_tau_i[meta_training_iter,task_ind,0,0:2] = bvarphi_tau_i_params[1].detach().numpy().copy()
            footprint_x_D_tau_tr[:] =    D_tau_tr.x.numpy()
            footprint_y_D_tau_tr[:,:] =  D_tau_tr.y.numpy()
            # Evaluate loss at the prior
            with torch.no_grad():
                kl_term =                   torch.tensor(0.0) # KL(xi,xi)=0 so xi.grad=0, so we can avoid calculating it
                ll_term = 				    torch.tensor(0.0) # to be accummulated
                for rr in range(numNets):
                    net_out =               net_meta_intermediate(  D_tau_tr.y, bvarphi_tau_i_params)
                    ll_term +=              criterion(  net_out.view(-1),
                                                        D_tau_tr.x      ) * len(D_tau_tr)   * BETA_MSE/2
                ll_term /= 					numNets
                loss_at_xi =                ll_term + KL_TEMP * kl_term
            footprint_task_loss   [meta_training_iter,task_ind,0    ] = loss_at_xi.detach().numpy().copy()
            for i in range(numSgdSteps): # m sgd steps over phi_tau_i. the last one is just for calc the loss
                if i==0 or bDegenerateToFrequentist:
                    kl_term =               torch.tensor(0.0) # KL(xi,xi)=0 so xi.grad=0, so we can avoid calculating it
                else:
                    kl_term =               KL_GaussianRandomNetworks( bvarphi_tau_i_params , bxi_params)
                ll_term = 				    torch.tensor(0.0) # to be accummulated
                for rr in range(numNets):
                    net_out =               net_meta_intermediate(  D_tau_tr.y, bvarphi_tau_i_params)
                    ll_term +=              criterion(  net_out.view(-1),
                                                        D_tau_tr.x      ) * len(D_tau_tr)   * BETA_MSE/2
                ll_term /= 					numNets
                local_loss =                ll_term + KL_TEMP * kl_term
                footprint_task_loss   [meta_training_iter,task_ind,i+1    ] = local_loss.detach().numpy().copy()
                local_grad =                torch.autograd.grad(    local_loss,
                                                                    KeepOnlyRequiredGrad(bvarphi_tau_i_params),
                                                                    create_graph= bCreateGraphForXi )
                                                                    # create_graph= {False: FOMAML , TRUE: MAML}
                bvarphi_tau_i_params =  list(map(   lambda p: p[1] - lr/len(D_tau_tr) * p[0],
                                                    zip(local_grad, KeepOnlyRequiredGrad(bvarphi_tau_i_params)))
                                             ) + deepcopy(KeepOnlyRequiredGradNot(bvarphi_tau_i_params))
                                                # adds no_grad params at the end. the new param_list has different id's
                                        # phi_tau_i_ <- phi_tau_i - lr * grad
                footprint_nu_tau_i    [meta_training_iter,task_ind,i+1,0:2] = bvarphi_tau_i_params[0].detach().numpy().copy()
                footprint_varrho_tau_i[meta_training_iter,task_ind,i+1,0:2] = bvarphi_tau_i_params[1].detach().numpy().copy()
            if (meta_training_iter==numMetaTrainingIter-1): # in last iter, save all per-task variational dist. params
                v_varphi_adapt.append(      [w.detach().clone() for w in bvarphi_tau_i_params])
            # calculate grad needed for meta-update. here bvarphi_tau_i_params not equal to bxi_params
            meta_loss_tau =         torch.tensor(0.0) # no KL for meta-loss, oblivious to bDegenerateToFrequentist
            rand_start = 			0 # torch.randperm(len(D_tau_te)-query_mini_batch_len)[0]
            query_dataset =         MyDataSet(  {   'y': D_tau_te.y[rand_start:(rand_start+query_mini_batch_len)] ,
                                                    'x': D_tau_te.x[rand_start:(rand_start+query_mini_batch_len)] })
            for rr in range(numNets):
                net_out =                   net_meta_intermediate(  query_dataset.y, bvarphi_tau_i_params)
                meta_loss_tau +=            criterion( net_out, query_dataset.x)   * len(query_dataset) * BETA_MSE/2
            meta_loss_tau /=                numNets
            vLoss_te[task_ind] =            meta_loss_tau.detach().clone()
            if (meta_training_iter % PRINT_INTERVAL == 0):
                print(print_str + outer_iter_str + inner_iter_str + f" meta_loss_tau={meta_loss_tau/query_mini_batch_len}")
            meta_grad_tau =                 torch.autograd.grad(    meta_loss_tau,
                                                                    KeepOnlyRequiredGrad(bxi_params),
                                                                    create_graph=False) # once we get meta-grad,
                                # we no longer need such dependencies. Therefore create_graph=False. True is also fine.
                                # accumulate meta-gradients w.r.t. tasks_batch
            for w,g in zip(KeepOnlyRequiredGrad(bxi_params), meta_grad_tau):
                w.total_grad +=         g.detach().clone()
            if (meta_training_iter % PRINT_INTERVAL == 0):
                channelstate =              l_mtr_channel_states[tau]
                print(print_str+ f'{outer_iter_str} nu{task_ind}={bvarphi_tau_i_params[0].detach().numpy()} exp(varrho{task_ind})={torch.exp(bvarphi_tau_i_params[1]).detach().numpy()} phiMMSE(c[{task_ind}])={channelstate_to_phi_mmse(channelstate,snr_dB).numpy()} phiLS[{task_ind}]={torch.inverse(D_tau_tr.y.t() @ D_tau_tr.y) @ (D_tau_tr.y.t() @ D_tau_tr.x).numpy()} loss_adapt[{task_ind}]={(vLoss_te[task_ind]/(len(query_dataset) * len(tasks_batch))).numpy()}')
        loss_adapt =                    torch.sum(vLoss_te) / (query_mini_batch_len*len(tasks_batch))
        vMetaTrainingLossAdapt.append(  loss_adapt.detach().clone()) # make a log of the meta-training loss
        if (loss_adapt <= minimal_training_loss_adapt):  # new loss is better
            print(print_str  + outer_iter_str + f"updating bxi_params_adapt having better loss {loss_adapt} <-------")
            minimal_training_loss_adapt =     loss_adapt.clone()  # update loss
            bxi_params_adapt =                deepcopy(bxi_params)  # update hyperparameters
            vMetaTrainingAdaptWithUpdates.append(meta_training_iter)
        if (meta_training_iter % PRINT_INTERVAL == 0):
            print(print_str+ f'{outer_iter_str} loss_adapt_on_te  = {loss_adapt.numpy()}')
        ## actual meta-update
        optimizer_meta.zero_grad()
        for w in KeepOnlyRequiredGrad(bxi_params):
            w.grad = w.total_grad.clone() /(query_mini_batch_len * tasks_batch_size_mtr)
        # updating bxi_params
        if  meta_training_iter < numMetaTrainingIter-1: # exclude the last run, for v_varphi to match xi
            # updating bxi_params
            optimizer_meta.step()   # bxi_params <- bxi_params - lr * meta_grad   ; only nu is updated (varrho has no_grad)
            #bxi_params = list(map(lambda p: p[1] - lr_shared / (query_mini_batch_len * tasks_batch_size_mtr) * p[0],
            #                    zip(meta_grad, KeepOnlyRequiredGrad(bxi_params)))
            #                ) + deepcopy(KeepOnlyRequiredGradNot(bxi_params))

    return bxi_params, \
           bxi_params_adapt, \
           {    'vMetaTrainingLossAdapt'        : vMetaTrainingLossAdapt,
                'vMetaTrainingAdaptWithUpdates' : vMetaTrainingAdaptWithUpdates,
                'vMetaTrainingTasksLoss'        : footprint_task_loss
           }, \
           v_varphi_adapt,\
           {    'nu_tau_i'                      : footprint_nu_tau_i,
                'varrho_tau_i'                  : footprint_varrho_tau_i,
                'phi_LS'                        : footprint_phi_LS,
                'phi_MRC'                       : footprint_phi_MRC,
                'phi_MMSE'                      : footprint_phi_MMSE,
                'Sigma_MAP_EBL'                 : footprint_Sigma_MAP_EBL,
                'nu_MAP_EBL'                    : footprint_nu_MAP_EBL,
                'x_D_tau_tr'                    : footprint_x_D_tau_tr,
                'y_D_tau_tr'                    : footprint_y_D_tau_tr
           }

def BaysianAdaptNnUsingGd(  bnet_params_init,
                            data_set_tr,
                            lr_first_sgd_steps          = 0.1, # will be reduced by 20 after numSgdSteps updates
                            numSgdSteps                 = 1,
                            numMetaAdaptIter            = 1,
                            num_first_samples_for_first_sgd_steps=None , # None means entire dataset for all iterations,
                            # if numbered, the first numSgdSteps updates will use a subset of data_set_tr using
                            # this many first samples
                            numNets                     = 100,
                            bDegenerateToFrequentist    = False, # {TRUE = Frequentist ; FALSE = Bayesian}
                            p2pSimo                     = None,  # p2pSimo (for modulation)
                            print_str                   = ''):
    if num_first_samples_for_first_sgd_steps is None:
        data_set_tr_to_use =        data_set_tr                 # use entire dataset from the beginning
    else:
        data_set_tr_to_use, _ =     data_set_tr.split_into_two_subsets(
                                                        num_first_samples_for_first_sgd_steps,      # numSamFirstSubSet
                                                        False )                                     # bShuffle=True
    vTrainingLoss =         np.zeros(numMetaAdaptIter)
    bnet_xi_for_kl_term =   deepcopy(bnet_params_init)
    bnet_adapted_params =   deepcopy(bnet_params_init)
    #bnet_optimal_params =   []
    #minimal_training_loss = +np.inf  # we rely on the training loss to avoid the need of test dataset while training
    criterion =             torch.nn.MSELoss()
    bnet_meta_intermediate =BayesianLinear_Meta() # for meta-learning; works with given NN params
    lr =                    lr_first_sgd_steps
    for meta_training_iter in range(numMetaAdaptIter):  # loop over the dataset multiple times
        iter_mad_str =          f" mad iter {meta_training_iter}/{numMetaAdaptIter}: "
        if bDegenerateToFrequentist:
            kl_term =       torch.tensor(0.0)       # no need of KL term, to be accumulated later
        else:
            kl_term =       KL_GaussianRandomNetworks( bnet_adapted_params ,
                                                       bnet_xi_for_kl_term) # to be accumulated
        ll_term =           torch.tensor(0.0)
        for rr in range(numNets):
            net_out =       bnet_meta_intermediate(data_set_tr_to_use.y, bnet_adapted_params)
            ll_term +=      criterion(  net_out,
                                        data_set_tr_to_use.x   )      * len(data_set_tr_to_use) * BETA_MSE/2
        ll_term /=          numNets
        loss =              ll_term + kl_term
        grad =              torch.autograd.grad(    loss,
                                                    KeepOnlyRequiredGrad(bnet_adapted_params),
                                                    create_graph = False)
        bnet_adapted_params = list(map(lambda p: p[1] - lr/len(data_set_tr_to_use) * p[0],
                                        zip(grad, KeepOnlyRequiredGrad(bnet_adapted_params)))
                                   ) + deepcopy(KeepOnlyRequiredGradNot(bnet_adapted_params))
                                # bnet_adapted_params<- bnet_adapted_params - lr*grad
        vTrainingLoss[meta_training_iter] = loss.detach().numpy()
        #with torch.no_grad():
        #    if bDegenerateToFrequentist:
        #        kl_term_after_epoch =           torch.tensor(0.0) # no need of KL term in freq., to be accumulated later
        #    else:
        #        kl_term_after_epoch =           KL_GaussianRandomNetworks( bnet_adapted_params,
        #                                                                   bnet_xi_for_kl_term) # to be accumulated
        #    ll_term_after_epoch =               torch.tensor(0.0)
        #    for rr in range(numNets):
        #        net_out =                       bnet_meta_intermediate(data_set_tr.y , bnet_adapted_params)
        #        ll_term_after_epoch +=          criterion(net_out, data_set_tr.x )  * len(data_set_tr)
        #    ll_term_after_epoch /=              numNets
        #    loss_after_epoch =                  ll_term_after_epoch + kl_term_after_epoch
        #if (loss_after_epoch <= minimal_training_loss):  # new loss is better
        #    minimal_training_loss =             loss_after_epoch.clone()  # update loss
        #    bnet_optimal_params = list(map(lambda p: deepcopy(p[0].data) , zip(bnet_adapted_params))) # workaround for deepcopy()
        #    print(print_str + iter_mad_str + f'new loss is the lowest {loss_after_epoch}')
        if meta_training_iter == numSgdSteps - 1:  # after m first iterations, make only gentle updates for the rest
            lr /=                               20                      # reduce learning rate dramatically
            data_set_tr_to_use =                data_set_tr             # use entire dataset from this point onwards
            bnet_xi_for_kl_term = list(map(lambda p: deepcopy(p[0].data), zip(bnet_adapted_params)))  # workaround for deepcopy()   !!!!!!!!!! p[0] ???
    return bnet_adapted_params, vTrainingLoss#bnet_optimal_params  # choose the best net we have seen so far in terms of training only

def BayesianMetaTest(varphi_params, dataset_star, numNets): # using only the inputs of dataset_star and not targets
    net_intermediate =              BayesianLinear_Meta()
    criterion =                     torch.nn.MSELoss()
    soft_out =                      torch.zeros(len(dataset_star.x))
                                            # size =  (no. of data points in set) X (no. of classes = len of last layer)
    with torch.no_grad():
        for rr in range(numNets):
            net_out =           net_intermediate(dataset_star.y, varphi_params)
            soft_out +=         net_out # torch.nn.functional.softmax(net_out, dim=1)
        soft_out /=             numNets # make sum into average
        #  this loss is indifferent of our assumed model with fixed precision beta, so the loss is the typical MSE loss
        mse =                   criterion(  dataset_star.x,
                                            soft_out    ) #* len(dataset_star) * BETA_MSE/2
        #ser =                   (xhat != dataset_star.x).numpy().mean()
    return mse # ser , soft_out

