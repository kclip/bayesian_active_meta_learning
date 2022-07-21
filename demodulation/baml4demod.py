
# algebra and matrices package
import numpy as np

import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
from scipy.io import loadmat, savemat

# Machine learning packages
from torch import nn
import torch
import torch.optim as optim

import datetime
from copy import deepcopy


REPARAM_COEFF =             0.1



def plot_constellation(iq, labels, desMod, titleStr):
    df = pd.DataFrame(dict(I=iq.ravel().real, Q=iq.ravel().imag, symbol=labels.ravel()))
    if sns.__version__ > '0.11':
        sns.jointplot(x='I', y='Q', data=df, hue='"symbol')     # use this instead when seaborn is upgraded to 11.1
    else:
        sns.lmplot(x = 'I', y = 'Q', data = df, hue = 'symbol', fit_reg = False)
    desMod.plot_decision_borders()  # draw borders for the desired modulation
    plt.title(titleStr)
    plt.grid()
    plt.axis('equal')
    plt.tight_layout()
    plt.draw()

def plot_curve_pm_width(vX, vYCurve, vYWidth, strColor='tab:blue', strLabel=''):
    plot_curve_with_boundaries(vX, np.vstack(vYCurve, vYCurve-vYWidth, vYCurve-vYWidth), strColor, strLabel)

def plot_curve_with_boundaries(vX, mY, strColor='tab:blue', strLabel=''): # mY.shape= [mX.shape,3] (lower,curve,upper)
    plt.plot        (vX, mY[:,1], '-', color=strColor,label=strLabel)
    plt.fill_between(vX, mY[:,0], mY[:,2], color=strColor, alpha=0.2)

def xconjx(x): # squared l2 norm for complex ndarrays
    return x.real*x.real + x.imag*x.imag

def SerOfSoftDemodNn(phi, dataset):
    with torch.no_grad():
        net_out =       phi(dataset.y)
    soft_out =          torch.nn.functional.softmax(net_out, dim=1)
    xhat =              torch.argmax(soft_out, dim = 1)
    ser =               (xhat != dataset.x).numpy().mean()
    return ser,  soft_out

def EceOfSoftDemodNn(soft_predictors, dataset_labels):  # Expected Calibration Error (see Guo2019)
    # dataset_labels should be input as dataset.x
    # soft_predictors should sum up to 1, so softmax should be used outside this function when needed.
    # Use case for freq. nn outputting logits:      EceOfSoftDemodNn( torch.softmax( phi(dataset.y,  dim = 1) , dataset)
    phat_i, xhat_i =    torch.max(soft_predictors.detach(), dim=1)
    vIndices =          np.arange(len(dataset_labels), dtype=int)
    M =                 10  # num of bins across (0,1]
    vAcc =              np.empty(shape=(M))  # M-len vector of accuracy   over M bins
    vConf =             np.empty(shape=(M))  # M-len vector of confidence over M bins
    vLenB_m =           np.empty(shape=(M), dtype=int)  # M-len vector of num of samples within each bin
    ece =               np.array(0.0)
    for m in range(M):  # m=0,1,...M-1
        B_m =           vIndices[(phat_i > 1.0 * m / M) & (phat_i <= (m + 1.0) / M)]
        vLenB_m[m] = len(B_m)
        if len(B_m) > 0:
            vAcc[m] =   (dataset_labels[B_m] == xhat_i[B_m]).numpy().sum() / len(B_m)
            vConf[m] =  (phat_i[B_m].numpy().sum()) / len(B_m)
            ece +=      abs(vAcc[m] - vConf[m]) * len(B_m) / len(dataset_labels)
        else:
            vAcc[m] =   np.nan
            vConf[m] =  np.nan
            # ece update is not needed for empty bin
    reliability_diagram = {"vAcc"           : vAcc,
                           "vConf"          : vConf,
                           "vBinCenters"    : np.linspace(1 / (2 * M), 1 - 1 / (2 * M), M),
                           "vLenB_m"        : vLenB_m
                           }
    return ece, reliability_diagram


class Modulation:
    def __init__(self, modKey):
        self.modKey = modKey
        dOrders = {  'WGN': [],
                    'QPSK':  4,
                   '16QAM': 16,
                   '64QAM': 64 }
        dNormCoeffs = {  'WGN': [],
                        'QPSK': torch.tensor(1 /  2.0).sqrt(),
                       '16QAM': torch.tensor(1 / 10.0).sqrt(),
                       '64QAM': torch.tensor(1 / 42.0).sqrt() }
        dMappings = {  'WGN': [],
                      'QPSK': torch.tensor( [+1.+1.j, -1.+1.j, +1.-1.j, -1.-1.j  ],
                                            dtype = torch.complex64),
                     '16QAM': torch.tensor( [-3.-3.j, -3.+1.j, +1.+1.j, +1.-3.j, -3.+3.j, +3.+1.j, +1.-1.j, -1.-3.j,
                                             +3.+3.j, +3.-1.j, -1.-1.j, -1.+3.j, +3.-3.j, -3.-1.j, -1.+1.j, +1.+3.j, ],
                                            dtype = torch.complex64),
                     '64QAM': torch.tensor( [-7.-7.j, -5.-7.j, -3.-7.j, -1.-7.j, +1.-7.j, +3.-7.j, +5.-7.j, +7.-7.j,
                                             -7.-5.j, -5.-5.j, -3.-5.j, -1.-5.j, +1.-5.j, +3.-5.j, +5.-5.j, +7.-5.j,
                                             -7.-3.j, -5.-3.j, -3.-3.j, -1.-3.j, +1.-3.j, +3.-3.j, +5.-3.j, +7.-3.j,
                                             -7.-1.j, -5.-1.j, -3.-1.j, -1.-1.j, +1.-1.j, +3.-1.j, +5.-1.j, +7.-1.j,
                                             -7.+1.j, -5.+1.j, -3.+1.j, -1.+1.j, +1.+1.j, +3.+1.j, +5.+1.j, +7.+1.j,
                                             -7.+3.j, -5.+3.j, -3.+3.j, -1.+3.j, +1.+3.j, +3.+3.j, +5.+3.j, +7.+3.j,
                                             -7.+5.j, -5.+5.j, -3.+5.j, -1.+5.j, +1.+5.j, +3.+5.j, +5.+5.j, +7.+5.j,
                                             -7.+7.j, -5.+7.j, -3.+7.j, -1.+7.j, +1.+7.j, +3.+7.j, +5.+7.j, +7.+7.j  ],
                                            dtype = torch.complex64)}
        # temporarily 64QAM (not gray, simple meshgrid)! complete later 64QAM using
        # http://literature.cdn.keysight.com/litweb/pdf/ads2008/3gpplte/ads2008/LTE_Mapper_(Mapper).html
        self.order = dOrders[modKey]
        self.normCoeff = dNormCoeffs[modKey]
        self.vMapping = dMappings[modKey]

    def modulate(self,vSamplesUint):
        return self.normCoeff * self.vMapping[vSamplesUint]

    def step(self, numSamples , xPattern):
        if self.modKey == 'WGN':
            vSamplesUint = torch.zeros(numSamples)
            vSamplesIq = (       torch.randn(numSamples)
                          + 1j * torch.randn(numSamples)    ) / torch.sqrt(2.0)
        else:
            if xPattern==0:             # 0,1,2,3,0,1,2,3,0,...
                vSamplesUint =          (torch.arange(0,numSamples, dtype = torch.long)   %self.order)
            elif xPattern==1:           # 0,0,0,0,1,1,1,1,2,...
                vSamplesUint =          (torch.arange(0, numSamples, dtype = torch.long)//self.order)%self.order
            elif xPattern == 160:       # like Sangwoo's paper for I of 16QAM
                vSamplesUint =          0,0,2,2,0,3,2,1,3,3,1,1,3,0,1,2
            elif xPattern == 161:  # like Sangwoo's paper for Q of 16QAM
                vSamplesUint =          0,2,2,0,3,2,1,0,3,1,1,3,0,1,2,3
            else:  # some random balanced permutation of (0,1,...M-1,0,1,...M-1,...)
                vSamplesUint =          torch.randperm(numSamples) % self.order
            vSamplesIq =                self.modulate(vSamplesUint)
        return (vSamplesIq, vSamplesUint)

    def hard_demodulator(self, vRx_iq):
        vHardRxUint = torch.argmin(torch.abs(   ( (vRx_iq.real+1j*vRx_iq.imag)
                                                - self.normCoeff * self.vMapping.unsqueeze(0)  ) ) ,
                                    axis=0)
        return vHardRxUint

    def plot_decision_borders(self):
        if self.modKey == 'QPSK':
            plt.plot([-2, 2], [0, 0], 'k--')
            plt.plot([0, 0], [-2, 2], 'k--')
        elif self.modKey == '16QAM' or self.modKey == '64QAM':
            dBordersOneDim = {'16QAM': np.arange(-2, 2 + 1, 2) * 1 / np.sqrt(10),
                              '64QAM': np.arange(-6, 6 + 1, 2) * 1 / np.sqrt(42)}
            vBorders = dBordersOneDim[self.modKey]
            delta = vBorders[2] - vBorders[1]
            for i in vBorders:
                plt.plot([i, i], [vBorders[0] - delta, vBorders[-1] + delta], 'k--')
                plt.plot([vBorders[0] - delta, vBorders[-1] + delta], [i, i], 'k--')

class IotUplink:

    def __init__(self, dSetting):
        self.dSetting =                 dSetting
        self.modulator =                Modulation(dSetting['modKey'])
        self.snr_lin =                  10.0 ** (self.dSetting['snr_dB'] / 10)
        self.randbeta =                 torch.distributions.beta.Beta(5, 2)

    def draw_channel_state(self): # samples new state to be used later
        self.dChannelState = {
            'tx_amp_imbalance_factor':      self.randbeta.sample()*0.15, # in [0,0.15]
            'tx_phase_imbalance_factor':    self.randbeta.sample()*15*3.14159265/180 ,#in [0,15] deg
            'ch_mult_factor':               torch.randn(1, dtype = torch.cfloat) # no need for torch.tensor(0.5).sqrt()
            }

    def step(self, numSamples , bEnforcePattern, bNoiseFree = False):
        if bEnforcePattern:
            pattern =       0
        else:
            pattern =       -1
        txIq , txSym =      self.modulator.step(numSamples, pattern)
        epsilon =           self.dChannelState['tx_amp_imbalance_factor']
        cos_delta =         torch.cos(self.dChannelState['tx_phase_imbalance_factor'])
        sin_delta =         torch.sin(self.dChannelState['tx_phase_imbalance_factor'])
        txDistorted =       (      (1+epsilon)*(cos_delta * txIq.real - sin_delta * txIq.imag)
                             +1j * (1-epsilon)*(cos_delta * txIq.imag - sin_delta * txIq.real) ) # TX non-lin
        txRayleighed =      txDistorted * self.dChannelState['ch_mult_factor'] # complex mult
        if bNoiseFree:
            noise =         0
        else:
            noise  =        np.sqrt(0.5 /self.snr_lin)*  torch.randn(numSamples, dtype=torch.cfloat)
        rxIq =              txRayleighed + noise
        rxReal =            torch.tensor([[z.real,z.imag] for z in rxIq]) # complex to two-dim tensor
        return MyDataSet(   {       'y': rxReal,
                                    'x': txSym  } )

class IotUplink2uSimo: # non linear TX + 2 users SIMO with numAnt antennas forming a critically spaced ULA

    def __init__(self, dSetting):
        self.dSetting =                 dSetting # SNR, modulation for all users, numAnt
        self.modulator =                Modulation(dSetting['modKey'])
        self.snr_lin =                  10.0 ** (self.dSetting['snr_dB'] / 10)
        self.randbeta =                 torch.distributions.beta.Beta(5, 2)
        self.dSetting['numAnt'] =       2
        self.dSetting['normSpacing'] =  0.5 # d over lambda

    def draw_channel_state(self): # samples new state to be used later
        self.dChannelState0 = {
            'tx_amp_imbalance_factor':      self.randbeta.sample()*0.15, # in [0,0.15]
            'tx_phase_imbalance_factor':    self.randbeta.sample()*15*3.14159265/180 ,#in [0,15] deg
            'ch_mult_factor':               torch.randn(1, dtype = torch.cfloat), # no need for torch.tensor(0.5).sqrt()
            'ch_doa_rad':                   (-60+120*torch.rand(1)) * np.pi / 180
            }
        self.dChannelState1 = {
            'tx_amp_imbalance_factor': self.randbeta.sample() * 0.15,  # in [0,0.15]
            'tx_phase_imbalance_factor': self.randbeta.sample() * 15 * 3.14159265 / 180,  # in [0,15] deg
            'ch_mult_factor': torch.randn(1, dtype=torch.cfloat),  # no need for torch.tensor(0.5).sqrt()
            'ch_doa_rad': (-60 + 120 * torch.rand(1)) * 2 * np.pi / 180
        }

    def step(self, numSamples , bEnforcePattern, bNoiseFree = False):
        if bEnforcePattern:
            pattern0 =      160
            pattern1 =      161
        else:
            pattern0 =      -1 # random
            pattern1 =      -1 # random
        txIq0 , txSym0 =    self.modulator.step(numSamples, pattern0)
        txIq1,  txSym1 =    self.modulator.step(numSamples, pattern1)
        def txRayleighed(dChannelState, txIq):
            epsilon =       dChannelState['tx_amp_imbalance_factor']
            cos_delta =     torch.cos(dChannelState['tx_phase_imbalance_factor'])
            sin_delta =     torch.sin(dChannelState['tx_phase_imbalance_factor'])
            txDistorted =   (      (1+epsilon)*(cos_delta * txIq.real - sin_delta * txIq.imag)
                             +1j * (1-epsilon)*(cos_delta * txIq.imag - sin_delta * txIq.real) ) # TX non-lin
            return          (txDistorted.view(-1,1) *                                       # col vec [numSamples]
                             dChannelState['ch_mult_factor'] *                              # complex algebra
                             torch.exp(-1j * 2 * np.pi * 0.5                                # scalars
                                       * torch.arange(self.dSetting['numAnt'],dtype=float)  # row vector
                                       * torch.sin(dChannelState['ch_doa_rad'])) )          # scalar
        txRayleighed0 =     txRayleighed(self.dChannelState0, txIq0)
        txRayleighed1 =     txRayleighed(self.dChannelState1, txIq1)
        if bNoiseFree:
            noise =         0
        else:
            noise  =        np.sqrt(0.5 /self.snr_lin)*  torch.randn_like(txRayleighed0, dtype=torch.cfloat)
        rxIq =              txRayleighed0 + txRayleighed1 + noise
        rxReal =            torch.cat([torch.cat((z.real,z.imag)) for z in rxIq]) # complex to two-dim tensor
        return MyDataSet(   {       'y': rxReal,
                                    'x': txSym0  } ) # only the first user is of interest

def EqualizerLmmseH(y_tr, complex_x_tr, y_te, x_te , varNoiseOverVarChannel):
    # returns a data set {x_te,y_te} for which the y is equalized using LMMSE with the error being over channel gain h,
    # and all tx non linearities are disregarded from the model
    complex_y_tr =                  y_tr[:,0] + 1j * y_tr[:,1]
    complex_y_te =                  y_te[:,0] + 1j * y_te[:,1]
    C_yy =                          (    complex_x_tr.view(-1, 1)                     # autocovariance;
                                       @ complex_x_tr.view( 1,-1).conj()             # cMat = cCol matmul cVec
                                     +   varNoiseOverVarChannel * torch.eye(len(complex_y_tr))  )
    c_yh =                          complex_x_tr.view(-1, 1)                          # make this a cCol
    weight_lmmse_h =                torch.linalg.solve(C_yy , c_yh )                  # cVec = cMat mutmul cVec
    hhat_lmmse_h =                  weight_lmmse_h.view(1,-1).conj() @ complex_y_tr   # cScalar = cRow matmul cCol
    y_equalized =                   1/hhat_lmmse_h         *   complex_y_te           # cScalar left mult sample-wise
    return                          MyDataSet({     'y': torch.cat( (   y_equalized.real.view(-1,1),
                                                                        y_equalized.imag.view(-1,1) )
                                                                  , 1                               ),
                                                    'x': x_te                  })# after LMMSE

def EqualizerLmmseX(y_tr, complex_x_tr, y_te, x_te):
    # returns a data set {x_te,y_te} for which the y is equalized using LMMSE with the error being over the symbols x,
    # and all tx non linearities are disregarded from the model
    complex_y_tr =                  y_tr[:,0] + 1j * y_tr[:,1]
    complex_y_te =                  y_te[:,0] + 1j * y_te[:,1]
    r_yy =                          torch.mean(  complex_y_tr.real**2
                                               + complex_y_tr.imag**2  )
    r_yx =                          torch.mean(  complex_y_tr  * complex_x_tr.conj() ) # cScalar = rScalar * cScalar
    weight_lmmse_x =                1/r_yy * r_yx                                      # cScalar = rScalar * cScalar
    y_equalized =                   weight_lmmse_x.conj()  *   complex_y_te            # cScalar left mult sample-wise
    return                          MyDataSet({     'y': torch.cat( (   y_equalized.real.view(-1,1),
                                                                        y_equalized.imag.view(-1,1) )
                                                                  , 1                               ),
                                                    'x': x_te                  })# after LMMSE

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

class FcReluDnn(nn.Module):     # Fully-Connected ReLU Deep Neural Network
    # Constructor
    def __init__ ( self, vLayers ):
        super(FcReluDnn, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(vLayers, vLayers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward ( self, activation ):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.nn.functional.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

    def num_parameters( self ):
        return sum([torch.numel(w) for w in self.parameters()])

class FcReluDnn_Meta_VerA(nn.Module):     # Fully-Connected ReLU Deep Neural Network for meta-learning
    # version by Sangwoo
    # Constructor
    def __init__ ( self ):
        super(FcReluDnn_Meta_VerA, self).__init__() # no need to initialize nn since we are given with parameter list
    # Prediction
    def forward ( self, net_in , net_params):   # net_in is the input and output.
                                                # net_params is the nn parameter vector, needs to be extarcted as
                                                #  list(map(lambda p: p[0], zip(nn_net.parameters())))
        L = int(len(net_params)/2) # weight, bias -> L stands for number of layers
        idx_nn = 0
        for l in range(L):
            curr_layer_weight = net_params[idx_nn]
            curr_layer_bias =   net_params[idx_nn+1]
            if l < L - 1:
                net_in= torch.nn.functional.relu(torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias))
                idx_nn += 2 # weight and bias
            else:
                net_in = torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias)
        return net_in

class FcReluDnn_Meta(nn.Module):     # Fully-Connected ReLU Deep Neural Network for meta-learning
    # version by Kfir
    # Constructor
    def __init__ ( self ):
        super(FcReluDnn_Meta, self).__init__() # no need to initialize nn since we are given with parameter list
    # Prediction
    def forward ( self, net_in , net_params): # net_in is the input and output. net_params is the nn parameter vector
        L = int(len(net_params)/2) # 2 for weight+bias in the parameter list; L stands for number of layers
        for ll in range(L):
            curr_layer_weight = net_params[2*ll  ]
            curr_layer_bias =   net_params[2*ll+1]
            net_in =            torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias)
            if ll<L-1: # inner layer has ReLU activation, last layer doesn't. Its softmax is in the loss func
                net_in=         torch.nn.functional.relu(net_in)
        return net_in


def FreqMamlMetaTrainVerB(  # Kfir
                                net_init,                   # instance of nnModule
                                data_set_meta_train,
                                lr =                        0.1,
                                lr_shared =                 0.1,
                                tasks_batch_size_mtr =      20,
                                numSgdSteps =               1,
                                numPilotsForMetaTraining =  4,
                                numMetaTrainingIter =       10000,
                                bFirstOrderApprox =         False,  # TRUE = FOMAML; FALSE = MAML with Hessians
                                print_str =                 '',
                                ):
    startTime =                         datetime.datetime.now()
    minimal_training_loss_adapt =       +np.inf
    xi_adapt =                          deepcopy(net_init)
    xi =                                deepcopy(net_init)
    criterion =                         nn.CrossEntropyLoss()
    optimizer_meta =                    torch.optim.Adam    (xi.parameters(), lr = lr_shared)
    vMetaTrainingLossXi =               []
    vMetaTrainingLossAdapt =            []
    vMetaTrainingAdaptWithUpdates =     []
    net_meta_intermediate = FcReluDnn_Meta() # for meta-learning; works with given NN parameter vector
    torch.manual_seed(0)  # for reproducibility with other MAML schemes, value not important - just need to be uniform
    for meta_training_iter in range(numMetaTrainingIter):
        iter_str =                      f"iter {meta_training_iter}/{numMetaTrainingIter} "
        tasks_batch = torch.randperm(len(data_set_meta_train))[:tasks_batch_size_mtr].numpy() #draw tasks minibatch
        lD_te =                         []
        lengths_lD_tr =                 torch.zeros(tasks_batch_size_mtr)
        lengths_lD_te =                 torch.zeros(tasks_batch_size_mtr)
        vLoss_te =                      torch.zeros(tasks_batch_size_mtr)
        xi.zero_grad()
        xi_params_list = list(map(lambda p: p[0], zip(xi.parameters())))
        for w in xi.parameters():
            w.total_grad = torch.zeros_like(w)
        for task_ind,tau in enumerate(tasks_batch):
            xi.zero_grad()
            [ D_tau_tr , D_tau_te ] =   data_set_meta_train[tau].split_into_two_subsets(numPilotsForMetaTraining, False)
            lengths_lD_tr[task_ind] =   len(D_tau_tr)
            lengths_lD_te[task_ind] =   len(D_tau_te)
            lD_te.append(               D_tau_te)
            # local update, stemming from xi
            phi_tau_i_params_list =     xi_params_list # initialization, to be reassigned on later updates
            for i in range(numSgdSteps): # m sgd steps over phi_tau_i
                net_out =               net_meta_intermediate(D_tau_tr.y,phi_tau_i_params_list)# locally updated phi_tau
                local_loss =            criterion(net_out, D_tau_tr.x)
                local_grad =            torch.autograd.grad(    local_loss,
                                                                phi_tau_i_params_list,
                                                                create_graph= not bFirstOrderApprox)
                                                                # create_graph= {False: FOMAML , TRUE: MAML}
                phi_tau_i_params_list = list(map(   lambda p: p[1] - lr * p[0],  # phi_tau_i_ <- phi_tau_i - lr * grad
                                                    zip(local_grad, phi_tau_i_params_list)))
            # calculate grad needed for meta-update
            net_out =                   net_meta_intermediate(D_tau_te.y, phi_tau_i_params_list)
            meta_loss =                 criterion(net_out, D_tau_te.x)
            vLoss_te[task_ind] =        meta_loss.detach().clone() * lengths_lD_te[task_ind]
            meta_grad = torch.autograd.grad(meta_loss,
                                            xi_params_list,
                                            create_graph=False) # once we get meta-grad,
                                 # we no longer need such dependencies. Therefore create_graph=False. True is also fine.
            # accumulate meta-gradients w.r.t. tasks_batch
            for w,g in zip(xi.parameters(),meta_grad):
                w.total_grad +=         g.detach().clone()
        loss_adapt =                    1 / torch.sum(lengths_lD_te) * torch.sum(vLoss_te)
        vMetaTrainingLossAdapt.append(  loss_adapt.detach().clone()) # make a log of the meta-training loss
        if (loss_adapt <= minimal_training_loss_adapt):  # new loss is better
            print(print_str + iter_str + f"updating xi_adapt having lowest loss {loss_adapt} <-------")
            minimal_training_loss_adapt =     loss_adapt.clone()  # update loss
            xi_adapt =                  deepcopy(xi)  # update hyperparameters
            vMetaTrainingAdaptWithUpdates.append(meta_training_iter)
        ## actual meta-update
        optimizer_meta.zero_grad()
        for w in xi.parameters():
            w.grad = w.total_grad.clone() / torch.sum(lengths_lD_te)
        # updating xi
        optimizer_meta.step()
        # evaluating sum L(xi) over minibatch of tasks:
        with torch.no_grad():
            loss_after_iter =           torch.tensor(0.0)
            for task_ind,tau in enumerate(tasks_batch):
                net_out =               xi(lD_te[task_ind].y)
                loss_after_iter +=      lengths_lD_te[task_ind] * criterion (net_out,  lD_te[task_ind].x)
            loss_after_iter /=          torch.sum(lengths_lD_te)
        vMetaTrainingLossXi.append(     loss_after_iter.detach().clone()) # make a log of the meta-training loss
        if (meta_training_iter % 100 == 0):
            print(print_str + iter_str + f"meta-loss-xi={loss_after_iter} min-meta-loss-phi={minimal_training_loss_adapt}")
            print(print_str + iter_str + f"time from start of meta training { datetime.datetime.now() - startTime }")
    return xi, xi_adapt, {  'vMetaTrainingLossXi'            : vMetaTrainingLossXi,
                            'vMetaTrainingLossAdapt'         : vMetaTrainingLossAdapt,
                            'vMetaTrainingAdaptWithUpdates'  : vMetaTrainingAdaptWithUpdates
                          }

def AdaptNnUsingGd(         net_init,
                            data_set_tr,
                            lr_first_sgd_steps = 0.1, # will be reduces by 20 after numSgdSteps updates
                            numSgdSteps =1,
                            numMetaAdaptIter = 1,
                            num_first_samples_for_first_sgd_steps =None # None means entire dataset for all iterations,
                            # if numbered, the first numSgdSteps updates will use a subset of data_set_tr using
                            # this many first samples
                            ):
    if num_first_samples_for_first_sgd_steps is None:
        P =                 len(data_set_tr)        # use entire dataset from the beginning
    else:
        P =                 num_first_samples_for_first_sgd_steps
    vTrainingLoss =         []
    net_adapted =           deepcopy(net_init)
    net_optimal =           []
    minimal_training_loss = +np.inf  # we rely on the training loss to avoid the need of test dataset while training
    criterion =             nn.CrossEntropyLoss()
    net_adapted.requires_grad_(True)
    optimizer =             optim.SGD(          net_adapted.parameters(),  # GD over trained_net
                                                lr = lr_first_sgd_steps)  # will be lower after the first m updates
    for meta_training_iter in range(numMetaAdaptIter):  # loop over the dataset multiple times
        optimizer.zero_grad()
        net_out =                   net_adapted(data_set_tr.y[0:P])
        loss =                      criterion(net_out, data_set_tr.x[0:P]) # only over the subset of the first P!
        loss.backward()
        optimizer.step()    # This is  "net_adapted <- net_adapted - lr * grad_net_adapted" with GD
        vTrainingLoss.append(loss.detach().numpy())
        with torch.no_grad():
            net_out =                           net_adapted(data_set_tr.y)
            loss_after_epoch =                  criterion(net_out, data_set_tr.x) # loss over entire dataset
        if (loss_after_epoch <= minimal_training_loss):  # new loss is better
            minimal_training_loss =             loss_after_epoch.clone()  # update loss
            net_optimal =                       deepcopy(net_adapted)  # update net
        if meta_training_iter == numSgdSteps - 1:  # after m first iterations, make only gentle updates for the rest
            optimizer.param_groups[0]['lr'] =   lr_first_sgd_steps / 20
            P =                                 len(data_set_tr)        # use entire dataset from this point onwards
    return net_optimal  # choose the best net we have seen so far in terms of training only


def AdaptNnUsingAdam(                       net_init,
                                            data_set_tr,
                                            lr_first_sgd_steps = 0.1,
                                            numSgdSteps = 1,
                                            batch_size = 20,
                                            vMetaAdaptIter = [10,100],
                                            weight_decay = 0
                                            ):
    net_adapted =           deepcopy(net_init)
    criterion =             nn.CrossEntropyLoss()
    optimizer =             optim.Adam(                     net_adapted.parameters(),
                                                            lr =            lr_first_sgd_steps,
                                                            weight_decay =  weight_decay)
    dataLoaderTrain =       torch.utils.data.DataLoader(    dataset =       data_set_tr,
                                                            batch_size =    min(int(batch_size), len(data_set_tr)),
                                                            shuffle =       True)
    minimal_training_loss =     +np.inf  # we rely on the training loss to avoid the need of test dataset while training
    net_optimal =               []
    list_networks =             []
    vLossAdapt =                np.zeros((vMetaAdaptIter[-1]))
    network_current_index =     0
    meta_training_iter =        0
    while meta_training_iter<vMetaAdaptIter[-1]:
        for batch_idx, (y, x) in enumerate(dataLoaderTrain):
            optimizer.zero_grad()
            net_out =                           net_adapted(y)
            loss =                              criterion(net_out, x)
            loss.backward()
            optimizer.step()                    # update net_adapted
            with torch.no_grad():
                net_out =                       net_adapted(data_set_tr.y)
                loss_after_iter =               criterion(net_out, data_set_tr.x)
                vLossAdapt[meta_training_iter] = loss_after_iter.clone()
            if (loss_after_iter <= minimal_training_loss):  # new loss is better
                minimal_training_loss =         loss_after_iter.clone()  # update loss
                net_optimal =                   deepcopy(net_adapted)  # update net
            #if meta_training_iter == numSgdSteps - 1:  # after m first iterations, make only gentle updates for the rest
            #    optimizer.param_groups[0]['lr'] = lr_first_sgd_steps / 20
            meta_training_iter +=               1
            if meta_training_iter==vMetaAdaptIter[network_current_index]:
                list_networks.append(           deepcopy(net_optimal))
                network_current_index +=        1
            if meta_training_iter>=vMetaAdaptIter[-1]:
                break
    return list_networks,vLossAdapt  # choose the best net we have seen so far in terms of training only


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

class BayesianFcReluDnn_Meta(nn.Module):     # Fully-Connected ReLU Deep Neural Network for meta-learning
    # In case you wish to generate a different network per sample, loop over the dataset,
    # use "inputs[ind].view(1,-1)" for the forward pass and "targets[ind].view(-1)" for the loss function,
    # and accumulate the loss foe averaging across the dataset before autograd.
    # Constructor
    def __init__ ( self ):
        super(BayesianFcReluDnn_Meta, self).__init__() # no need to init since we are given with parameter list
    # Prediction
    def forward ( self, bnet_in , bnet_params): # net_in is the input and output. net_params is the nn parameter vector
        L = int(len(bnet_params) / 2 / 2) # first 2 by weight+bias, second 2 for nu+varrho L stands for number of layers
        for ll in range(L): ## 2*ll is the weight parameter, 2*ll+1 is the bias
            weight_nu =             bnet_params[      2*ll    ] # in-place assignment, same id() for LHS and RHS
            bias_nu =               bnet_params[      2*ll + 1] # "
            weight_varrho =         bnet_params[2*L + 2*ll    ] # "
            bias_varrho =           bnet_params[2*L + 2*ll + 1] # "
            weight_random =         weight_nu + torch.exp(weight_varrho) * torch.randn_like(weight_varrho)*REPARAM_COEFF
            bias_random =           bias_nu   + torch.exp(  bias_varrho) * torch.randn_like(  bias_varrho)*REPARAM_COEFF
            bnet_in =               torch.nn.functional.linear(bnet_in, weight_random, bias_random)
            if ll<L-1: # inner layer has ReLU activation, last layer doesn't. Its softmax is in the loss func
                bnet_in=            torch.nn.functional.relu(bnet_in)
        return bnet_in


def BayesianMamlMetaTrain(
                                bnet_params_init,                  # Bayesian net of init as list of parameters (mean+exp(std))
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
                                print_str =                 '',
                                ):
    # bxi_params_list should be by      (   list(map(lambda p: p[0], zip(nu.parameters()    )))
    #                                     + list(map(lambda p: p[0], zip(varrho.parameters())))      )
    startTime =                         datetime.datetime.now()
    minimal_training_loss_adapt =       +np.inf
    bxi_params_adapt =                  deepcopy(bnet_params_init)
    bxi_params =                        deepcopy(bnet_params_init)
    criterion =                         nn.CrossEntropyLoss()
    query_mini_batch_len =              160 # for meta updates
    optimizer_meta =                    torch.optim.Adam ( bxi_params, lr = lr_shared )
    vMetaTrainingLossXi =               []
    vMetaTrainingLossAdapt =            []
    vMetaTrainingAdaptWithUpdates =     []
    net_meta_intermediate =             BayesianFcReluDnn_Meta() # for single data point
    torch.manual_seed(0)  # for reproducibility with other MAML schemes, value not important - just need to be uniform
    for meta_training_iter in range(numMetaTrainingIter):
        iter_str =                      f"iter {meta_training_iter}/{numMetaTrainingIter} "
        tasks_batch = torch.randperm(len(data_set_meta_train))[:tasks_batch_size_mtr].numpy() #draw tasks minibatch
        lD_te =                         []
        lengths_lD_tr =                 torch.zeros(tasks_batch_size_mtr)
        lengths_lD_te =                 torch.zeros(tasks_batch_size_mtr)
        vLoss_te =                      torch.zeros(tasks_batch_size_mtr)
        for w in KeepOnlyRequiredGrad(bxi_params):
            w.total_grad = torch.zeros_like(w)
        for task_ind,tau in enumerate(tasks_batch):
            task_str =                  f"task_ind={task_ind} "
            [ D_tau_tr , D_tau_te ] =   data_set_meta_train[tau].split_into_two_subsets(numPilotsForMetaTraining, False)
            lengths_lD_tr[task_ind] =   len(D_tau_tr)
            lengths_lD_te[task_ind] =   len(D_tau_te)
            lD_te.append(               D_tau_te)
            # local update, stemming from bxi_params
            bvarphi_tau_i_params =      bxi_params # initialization, to be reassigned on later updates.
            # no need to zero the grad of the LHS since the RHS comes with grad=0 and the assignment is a copied version
            for i in range(numSgdSteps): # m sgd steps over phi_tau_i
                if i==0 or bDegenerateToFrequentist:
                    kl_term =               torch.tensor(0.0) # KL(xi,xi)=0 so xi.grad=0, so we can avoid calculating it
                else:
                    kl_term =               KL_GaussianRandomNetworks( bvarphi_tau_i_params , bxi_params)
                ll_term = 				    torch.tensor(0.0) # to be accummulated
                for rr in range(numNets):
                    net_out =               net_meta_intermediate(  D_tau_tr.y, bvarphi_tau_i_params)
                    ll_term +=              criterion(  net_out,       D_tau_tr.x)     * len(D_tau_tr)
                ll_term /= 					numNets
                local_loss =                ll_term + kl_term
                local_grad =                torch.autograd.grad(    local_loss,
                                                                    KeepOnlyRequiredGrad(bvarphi_tau_i_params),
                                                                    create_graph= not bFirstOrderApprox)
                                                                    # create_graph= {False: FOMAML , TRUE: MAML}
                bvarphi_tau_i_params =  list(map(   lambda p: p[1] - lr/len(D_tau_tr) * p[0],
                                                    zip(local_grad, KeepOnlyRequiredGrad(bvarphi_tau_i_params)))
                                             ) + deepcopy(KeepOnlyRequiredGradNot(bvarphi_tau_i_params))
                                                # adds no_grad params at the end. the new param_list has different id's
                                        # phi_tau_i_ <- phi_tau_i - lr * grad
            # calculate grad needed for meta-update. here bvarphi_tau_i_params not equal to bxi_params
            meta_loss =             torch.tensor(0.0) # no KL for meta-loss, oblivious to bDegenerateToFrequentist
            rand_start = 			torch.randperm(len(D_tau_te)-query_mini_batch_len)[0]
            query_dataset =         MyDataSet(  {   'y': D_tau_te.y[rand_start:(rand_start+query_mini_batch_len)] ,
                                                    'x': D_tau_te.x[rand_start:(rand_start+query_mini_batch_len)] })
            soft_out = 	    		torch.zeros(len(query_dataset),len(bvarphi_tau_i_params[-1]))
                                    # dim= [len(query_dataset) , num of classes ]
            for rr in range(numNets):
                net_out =                   net_meta_intermediate(  query_dataset.y, bvarphi_tau_i_params)
                meta_loss +=                criterion( net_out, query_dataset.x)   * len(query_dataset)
                soft_out +=         		torch.nn.functional.softmax(net_out, dim=1)
            soft_out /=                     numNets
            xhat =                          torch.argmax(soft_out, dim=1)
            ser =                           (xhat != query_dataset.x).numpy().mean()
            vLoss_te[task_ind] =            meta_loss.detach().clone()
            print(print_str + iter_str + task_str + f" meta_loss_tau={meta_loss/query_mini_batch_len} ser={ser}")
            meta_grad =                     torch.autograd.grad(    meta_loss,
                                                                    KeepOnlyRequiredGrad(bxi_params),
                                                                    create_graph=False) # once we get meta-grad,
                                 # we no longer need such dependencies. Therefore create_graph=False. True is also fine.
            # accumulate meta-gradients w.r.t. tasks_batch
            for w,g in zip(KeepOnlyRequiredGrad(bxi_params), meta_grad):
                w.total_grad +=         g.detach().clone()
        loss_adapt =                    torch.sum(vLoss_te) / (query_mini_batch_len*tasks_batch_size_mtr)
        vMetaTrainingLossAdapt.append(  loss_adapt.detach().clone()) # make a log of the meta-training loss
        if (loss_adapt <= minimal_training_loss_adapt):  # new loss is better
            print(print_str  + iter_str + f"updating bxi_params_adapt having better loss {loss_adapt} <-------")
            minimal_training_loss_adapt =     loss_adapt.clone()  # update loss
            bxi_params_adapt =                deepcopy(bxi_params)  # update hyperparameters
            vMetaTrainingAdaptWithUpdates.append(meta_training_iter)
        ## actual meta-update
        optimizer_meta.zero_grad()
        for w in KeepOnlyRequiredGrad(bxi_params):
            w.grad = w.total_grad.clone() /(query_mini_batch_len*tasks_batch_size_mtr)
        # updating bxi_params
        optimizer_meta.step()   # bxi_params <- bxi_params - lr * meta_grad   ; only nu is updated (varrho has no_grad)
        # evaluating sum_tau{L(bxi_params){ over minibatch of tasks:
        with torch.no_grad():
            loss_after_iter =           torch.tensor(0.0)
            for task_ind,tau in enumerate(tasks_batch):
                # since KL_GaussianRandomNetworks(bxi_params, bxi_params)=0, no need to recalculate it
                net_out =               net_meta_intermediate( lD_te[task_ind].y, bxi_params)
                loss_after_iter +=      criterion(             net_out,       lD_te[task_ind].x) * len(lD_te[task_ind])
            loss_after_iter /=          torch.sum(lengths_lD_te)
        vMetaTrainingLossXi.append(     loss_after_iter.detach().clone()) # make a log of the meta-training loss
        if (meta_training_iter % 1 == 0):
            print(print_str + iter_str + f"meta-loss-xi={loss_after_iter} min-meta-loss-phi={minimal_training_loss_adapt}")
            print(f"time from start of meta training { datetime.datetime.now() - startTime }")
    return bxi_params, bxi_params_adapt, {  'vMetaTrainingLossXi'            : vMetaTrainingLossXi,
                                            'vMetaTrainingLossAdapt'         : vMetaTrainingLossAdapt,
                                            'vMetaTrainingAdaptWithUpdates'  : vMetaTrainingAdaptWithUpdates
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
                            print_str                   = ''):
    if num_first_samples_for_first_sgd_steps is None:
        data_set_tr_to_use =        data_set_tr                 # use entire dataset from the beginning
    else:
        data_set_tr_to_use, _ =     data_set_tr.split_into_two_subsets(
                                                        num_first_samples_for_first_sgd_steps,      # numSamFirstSubSet
                                                        False )                                     # bShuffle=True
    vTrainingLoss =         []
    bnet_xi_for_kl_term =   deepcopy(bnet_params_init)
    bnet_adapted_params =   deepcopy(bnet_params_init)
    bnet_optimal_params =   []
    minimal_training_loss = +np.inf  # we rely on the training loss to avoid the need of test dataset while training
    criterion =             nn.CrossEntropyLoss()
    bnet_meta_intermediate =BayesianFcReluDnn_Meta() # for meta-learning; works with given NN params
    lr =                    lr_first_sgd_steps
    for meta_training_iter in range(numMetaAdaptIter):  # loop over the dataset multiple times
        iter_str =          f"{meta_training_iter}/{numMetaAdaptIter}: "
        if bDegenerateToFrequentist:
            kl_term =       torch.tensor(0.0)       # no need
            # of KL term, to be accumulated later
        else:
            kl_term =       KL_GaussianRandomNetworks( bnet_adapted_params ,
                                                       bnet_xi_for_kl_term) # to be accumulated
        ll_term =           torch.tensor(0.0)
        for rr in range(numNets):
            net_out =       bnet_meta_intermediate(data_set_tr_to_use.y, bnet_adapted_params)
            ll_term +=      criterion(net_out, data_set_tr_to_use.x )  * len(data_set_tr_to_use)
        ll_term /=          numNets
        loss =              ll_term + kl_term
        grad =              torch.autograd.grad(    loss,
                                                    KeepOnlyRequiredGrad(bnet_adapted_params),
                                                    create_graph = False)
        bnet_adapted_params = list(map(lambda p: p[1] - lr/len(data_set_tr_to_use) * p[0],
                                        zip(grad, KeepOnlyRequiredGrad(bnet_adapted_params)))
                                   ) + deepcopy(KeepOnlyRequiredGradNot(bnet_adapted_params))
                                # bnet_adapted_params<- bnet_adapted_params - lr*grad
        vTrainingLoss.append(loss.detach().numpy())
        with torch.no_grad():
            if bDegenerateToFrequentist:
                kl_term_after_epoch =           torch.tensor(0.0) # no need of KL term in freq., to be accumulated later
            else:
                kl_term_after_epoch =           KL_GaussianRandomNetworks( bnet_adapted_params,
                                                                           bnet_xi_for_kl_term) # to be accumulated
            ll_term_after_epoch =               torch.tensor(0.0)
            for rr in range(numNets):
                net_out =                       bnet_meta_intermediate(data_set_tr.y , bnet_adapted_params)
                ll_term_after_epoch +=          criterion(net_out, data_set_tr.x )  * len(data_set_tr)
            ll_term_after_epoch /=              numNets
            loss_after_epoch =                  ll_term_after_epoch + kl_term_after_epoch
        if (loss_after_epoch <= minimal_training_loss):  # new loss is better
            minimal_training_loss =             loss_after_epoch.clone()  # update loss
            bnet_optimal_params = list(map(lambda p: deepcopy(p[0].data) , zip(bnet_adapted_params))) # workaround for deepcopy()
            print(print_str+iter_str+f'new loss is the lowest {loss_after_epoch}')
        if meta_training_iter == numSgdSteps - 1:  # after m first iterations, make only gentle updates for the rest
            lr /=                               20                      # reduce learning rate dramatically
            data_set_tr_to_use =                data_set_tr             # use entire dataset from this point onwards
            bnet_xi_for_kl_term = list(map(lambda p: deepcopy(p[0].data), zip(bnet_adapted_params)))  # workaround for deepcopy()   !!!!!!!!!! p[0] ???
    return bnet_optimal_params  # choose the best net we have seen so far in terms of training only

def BayesianMetaTest(varphi_params, dataset_star, numNets): # using only the inputs of dataset_star and not targets
    net_intermediate =              BayesianFcReluDnn_Meta()
    soft_out =                      torch.zeros(len(dataset_star.x),len(varphi_params[-1]))
                                            # size =  (no. of data points in set) X (no. of classes = len of last layer)
    with torch.no_grad():
        for rr in range(numNets):
            net_out =           net_intermediate(dataset_star.y, varphi_params)
            soft_out +=         torch.nn.functional.softmax(net_out, dim=1)
        soft_out /=             numNets # make sum into average
        xhat =                  torch.argmax(soft_out, dim = 1)
        ser =                   (xhat != dataset_star.x).numpy().mean()
    return ser , soft_out

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

