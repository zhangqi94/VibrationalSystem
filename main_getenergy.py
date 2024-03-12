import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import time
import os

####################################################################################
print("jax.__version__:", jax.__version__)
import subprocess
print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True).stdout, flush=True)
key = jax.random.PRNGKey(42)

####################################################################################
"""
srun -p v100 --pty --gres=gpu:1 -t 0-9999:00 /bin/bash
srun -p a800 --pty --gres=gpu:1 -t 0-9999:00 /bin/bash
srun -p a100 -N 1 --gres=gpu:A100_40G:1 --pty /bin/bash
srun -p a100 -N 1 --gres=gpu:A100_80G:1 --pty /bin/bash

.venv/bin/python3.11 ./main_getenergy.py  \
    --file_name "/data/ruisiwang/Vibration_gitlab_repo/CH3CN/CH3CN_n_12_alp_1000_orb_84_rnvp_16_mlp_128_2_mcthr_20_stp_200_std_0.2_adamlr_0.001_clp_5_bth_1024_acc_1_nde_1/epoch_008300.pkl"\
    --num_devices=1  --batch=1024  --acc_steps=1  --mc_therm=20  --mc_steps=200  --mc_stddev=0.2 \
    --cal_orb 0 1 3 4 6 7 10 11 9 13 15 16 17 18 20 25 56 55 79 82

.venv/bin/python3.11 ./main_getenergy.py  \
    --file_name "/data/ruisiwang/Vibration_gitlab_repo/CH3CN/CH3CN_n_12_alp_1000_orb_84_rnvp_16_mlp_256_2_mcthr_20_stp_200_std_0.2_adamlr_0.001_clp_5_bth_1024_acc_1_nde_1/epoch_004000.pkl"\
    --num_devices=1  --batch=1024  --acc_steps=1  --mc_therm=20  --mc_steps=200  --mc_stddev=0.2 \
    --cal_orb 0 1 3 4 6 7 10 11 9 13 15 16 17 18 20 25 56 55 79 82

.venv/bin/python3.11 ./main_getenergy.py  \
    --file_name "/data/ruisiwang/Vibration_gitlab_repo/CH3CN/CH3CN_n_12_alp_1000_orb_84_rnvp_32_mlp_128_2_mcthr_20_stp_200_std_0.2_adamlr_0.001_clp_5_bth_1024_acc_1_nde_1/epoch_004200.pkl"\
    --num_devices=1  --batch=1024  --acc_steps=1  --mc_therm=20  --mc_steps=200  --mc_stddev=0.2 \
    --cal_orb 0 1 3 4 6 7 10 11 9 13 15 16 17 18 20 25 56 55 79 82
"""
####################################################################################
#========== file name ==========
import argparse
parser = argparse.ArgumentParser(description="calculate exicted energy")
parser.add_argument("--file_name", default="", 
                    help="the folder to save data")
parser.add_argument("--num_devices", default=1, type=int, help="number of GPU devices")
parser.add_argument("--cal_orb", type=int, nargs="+", default=None, help="choose orbital")
parser.add_argument("--batch", default=65536, type=int, help="batch size")
parser.add_argument("--acc_steps", default=16, type=int, help="number of accumulation steps")
parser.add_argument("--mc_therm", default=20, type=int, help="number of thermal steps")
parser.add_argument("--mc_steps", default=500, type=int, help="number of MCMC steps")
parser.add_argument("--mc_stddev", default=0.2, type=float, help="standard deviation of MCMC")

args = parser.parse_args()
file_name = args.file_name
num_devices = args.num_devices
cal_orb = args.cal_orb

#========== load file ==========
from src.checkpoint import load_data
datas = load_data(file_name)

mode, w2_indices, lam = datas["mode"], datas["w2_indices"], datas["lam"]
n, dim, num_orb, alpha = datas["n"], datas["dim"], datas["num_orb"], datas["alpha"]
flow_type = datas["flow_type"]
flow_depth, mlp_width, mlp_depth = datas["flow_depth"], datas["mlp_width"], datas["mlp_depth"]
dsf_width, dsf_depth = datas["dsf_width"], datas["dsf_depth"]


params_flow = datas["params_flow"]

#batch, acc_steps = 8192*16, 8
#mc_therm, mc_steps, mc_stddev = 20, 500, 0.2
batch, acc_steps = args.batch, args.acc_steps
mc_therm, mc_steps, mc_stddev = args.mc_therm, args.mc_steps, args.mc_stddev

hutchinson = False
epoch = 1

print("mode:", mode, "  w2_indices:", w2_indices, "  lam:", lam)
print("n:", n, "  dim:", dim, "  alpha:", alpha)
print("flow_type:", flow_type)
print("flow_depth:", flow_depth, "  mlp_width:", mlp_width, "  mlp_depth:", mlp_depth)
if flow_type == "NAF": 
    print("dsf_width:", dsf_width, "  dsf_depth:", dsf_depth)
print("mc_therm:", mc_therm, "  mc_steps:", mc_steps, "  mc_stddev:", mc_stddev)
print("batch:", batch, "  acc_steps:", acc_steps, "  num_devices:", num_devices)
print("mc_therm:", mc_therm, "  mc_steps:", mc_steps, "  mc_stddev:", mc_stddev)      
      
####################################################################################
print("\n========== Initialize Hamiltonian ==========")
print("Harmonic oscillator:\n    mode: %s, n: %d, dim: %d" % (mode, n, dim), flush=True)

# mode type: H2O, C2H4O, H2CO, NDCO, mode-2 (toy model), mode-3 (toy model)
if mode == "H2O":
    from src.potential import get_potential_energy_water
    potential_energy, w_indices = get_potential_energy_water(alpha=alpha)
elif mode == "H2CO":
    from src.potential import get_potential_energy_H2CO
    potential_energy, w_indices = get_potential_energy_H2CO(alpha=alpha)
elif mode == "C2H4O":
    from src.potential import get_potential_energy_C2H4O
    potential_energy, w_indices = get_potential_energy_C2H4O(alpha=alpha)
elif mode == "NDCO": # N-dimensional coupled oscillator
    from src.potential import get_potential_energy_NDCO, calculate_exact_energy_NDCO
    potential_energy, w_indices = get_potential_energy_NDCO(D=n)
elif mode == "TOY2": # two-mode system
    from src.potential import get_potential_energy_mode2
    potential_energy, w_indices = get_potential_energy_mode2(lam, w2_indices)
elif mode == "TOY3": # three-mode system
    from src.potential import get_potential_energy_mode3
    potential_energy, w_indices = get_potential_energy_mode3(lam, w2_indices)    
elif mode == "CH3CN":
    from src.potential import get_potential_energy_CH3CN
    potential_energy, w_indices = get_potential_energy_CH3CN(alpha=alpha)

if len(w_indices) != n: raise ValueError("Number of modes is not consistent!")
invsqrtw = 1/jnp.sqrt(w_indices)

####################################################################################
print("\n========== Initialize orbitals ==========")
from src.potential.orbitals import get_orbitals_1d, get_orbitals_indices_first, orbitals_array2str
if dim != 1: raise ValueError("Only dim = 1 is supported!")
sp_orbitals, _ = get_orbitals_1d()

orb_index, orb_state, orb_Es = datas["orb_index"], datas["orb_state"], datas["orb_Es"]

print("Total number of orbitals:", num_orb)
if mode == "NDCO": # calculate exact energy of NDCO
    orb_index, orb_state, orb_Es, nu = calculate_exact_energy_NDCO(D=n, num_orb=num_orb)
    print("nu:", nu, "\nExact energy of each energy level:")
    for ii in range(num_orb): 
        print("    %d, E: %.12f, level:" %(ii, orb_Es[ii]), orbitals_array2str(orb_state[ii]), flush=True)  
elif mode == "TOY2" or mode == "TOY3": # toy model, mode2 or mode3 
    print("Energy of non-interacting system:")
    for ii in range(num_orb):
        print("    %d, E: %.12f, level:" %(ii, orb_Es[ii]), orb_state[ii], flush=True)
else: ## H2O, C2H4O, H2CO
    print("Energy of non-interacting system:")
    for ii in range(num_orb):  
        print("    %d, E: %.2f, level:" %(ii, orb_Es[ii]), orbitals_array2str(orb_state[ii]), flush=True)

####################################################################################
print("\n========== Initialize flow model & logpsi_wavefunction ==========")
if flow_type == "NAF":
    import src.flow_NAF
    flow = src.flow_NAF.make_flow(key, flow_depth, mlp_width, mlp_depth, dsf_width, dsf_depth, n*dim)
    print("NAF:    depth: %d" % flow_depth, ",  mlp hidden layers: %d, %d" %(mlp_width, mlp_depth))
    print("DSF:    width: %d" % dsf_width, ",  depth: %d" % dsf_depth)
if flow_type == "RNVP":
    import src.flow_RNVP
    flow = src.flow_RNVP.make_flow(key, flow_depth, mlp_width, mlp_depth, n*dim)
    print("RNVP:    depth: %d" % flow_depth, ",  mlp hidden layers: %d, %d" %(mlp_width, mlp_depth))

params_flow = datas["params_flow"] # load params of flow from file
raveled_params_flow, _ = ravel_pytree(params_flow)
print("    parameters in the flow model: %d" % raveled_params_flow.size, flush=True)

#========== logpsi = logphi + 0.5*logjacdet ==========
import src.logpsi
logpsi_novmap = src.logpsi.make_logpsi(flow, sp_orbitals, orb_state, w_indices)
logphi, logjacdet = src.logpsi.make_logphi_logjacdet(flow, sp_orbitals, orb_state, w_indices)
logp = src.logpsi.make_logp(logpsi_novmap)

####################################################################################
print("\n========== Check point ==========")
from src.utils import replicate, shard
import src.checkpoint

#========== initialize ==========
print("Number of GPU devices:", num_devices)
if num_devices != jax.device_count():
    raise ValueError("Expected %d GPU devices. Got %d." % (num_devices, jax.device_count()))

if batch % num_devices != 0:
    raise ValueError("Batch size must be divisible by the number of GPU devices. "
                        "Got batch = %d for %d devices now." % (batch, num_devices))
batch_per_device = batch // num_devices

x = jax.random.normal(key, (num_devices, batch_per_device, n, dim))
keys = jax.random.split(key, num_devices)
x, keys = shard(x), shard(keys)
params_flow = replicate(params_flow, num_devices)
print("x.shape:", x.shape)

#========== grad, laplacian ==========
logpsi, logpsi_grad_laplacian = src.logpsi.make_logpsi_grad_laplacian(logpsi_novmap, forloop=True,
                                        hutchinson=hutchinson, logphi=logphi, logjacdet=logjacdet)

#========== observable, loss function ==========
import src.VMCobs
observable_and_lossfn = src.VMCobs.make_loss(logpsi, logpsi_grad_laplacian, potential_energy)

#========== update function ==========
from functools import partial
@partial(jax.pmap, axis_name="p",
        in_axes=(0, 0, 0, 0, 0, None),
        out_axes=(0),
        static_broadcasted_argnums=(5),
        donate_argnums=(2))
def update(params_flow, state_indices, x, key, data_acc, final_step):
    data = observable_and_lossfn(params_flow, state_indices, x, key)
    data_acc = jax.tree_map(lambda acc, i: acc + i, (data_acc), (data))
    if final_step:
        data_acc = jax.tree_map(lambda acc: acc / acc_steps, (data_acc))
    return data_acc

####################################################################################
print("\n========== Energy levels measured ==========")
if cal_orb != None:
    cal_orb = np.array(cal_orb, dtype=np.int64)
    cal_n_orbital = cal_orb
else:
    cal_n_orbital = range(num_orb)
print("Calculate these energy levels: (Energies for non-interacting systems)")
for ii in cal_n_orbital:  
    print("    %.3d, E: %.6f, level:" %(ii, orb_Es[ii]), orbitals_array2str(orb_state[ii]), flush=True)

####################################################################################
print("\n========== Start measuring ==========")

list_E, list_Estd = np.zeros((num_orb,), dtype=np.float64), np.zeros((num_orb,), dtype=np.float64)
list_K, list_Kstd = np.zeros((num_orb,), dtype=np.float64), np.zeros((num_orb,), dtype=np.float64)
list_V, list_Vstd = np.zeros((num_orb,), dtype=np.float64), np.zeros((num_orb,), dtype=np.float64)
#========== circulate ==========
for n_orbital in cal_n_orbital:
    tstart = time.time()
    index = jnp.array([n_orbital])
    state_indices = jnp.tile(index, (num_devices, batch_per_device))
    state_indices = shard(state_indices)

    print("\n"+"="*80)
    for ii in range(mc_therm):
        t0 = time.time()
        keys, x, accept_rate = src.VMCobs.sample_x_mcmc(keys, state_indices,
                                   logp, x, params_flow, mc_steps, mc_stddev, invsqrtw)
        t1 = time.time()
        print("---- thermal step: %d,  ac: %.4f,  dx: %.4f,  dt: %.3f ----" 
                % (ii+1, accept_rate[0], mc_stddev, t1-t0), flush=True)
        if accept_rate[0] > 0.525: mc_stddev *= 1.1
        if accept_rate[0] < 0.475: mc_stddev *= 0.9
        
    for ii in range(1, epoch + 1):
        data_acc = replicate({"E_mean": 0., "E2_mean": 0., "K_mean": 0., "K2_mean": 0.,
                              "V_mean": 0., "V2_mean": 0.,}, num_devices)

        accept_rate_acc = shard(jnp.zeros(num_devices))
        batch_per_device = batch // num_devices

        for acc in range(acc_steps):
            t0 = time.time()
            keys, x, accept_rate = src.VMCobs.sample_x_mcmc(keys, state_indices,
                                        logp, x, params_flow, mc_steps, mc_stddev, invsqrtw)
            accept_rate_acc += accept_rate
            final_step = (acc == (acc_steps-1))
            data_acc = update(params_flow, state_indices, x, keys, data_acc, final_step)
            t1 = time.time()
            print("---- accept step: %d,  ac: %.4f,  dx: %.4f,  dt: %.3f ----" 
                    % (acc+1, accept_rate[0], mc_stddev, t1-t0), flush=True)
            
        data = jax.tree_map(lambda x: x[0], data_acc)
        accept_rate = accept_rate_acc[0] / acc_steps
        
        E, E2_mean = data["E_mean"]*alpha, data["E2_mean"]*(alpha**2)
        K, K2_mean = data["K_mean"]*alpha, data["K2_mean"]*(alpha**2)
        V, V2_mean = data["V_mean"]*alpha, data["V2_mean"]*(alpha**2)
        E_std = jnp.sqrt((E2_mean - E**2) / (batch*acc_steps))
        K_std = jnp.sqrt((K2_mean - K**2) / (batch*acc_steps))
        V_std = jnp.sqrt((V2_mean - V**2) / (batch*acc_steps))
        
        list_E[n_orbital], list_Estd[n_orbital] = E, E_std
        list_K[n_orbital], list_Kstd[n_orbital] = K, K_std
        list_V[n_orbital], list_Vstd[n_orbital] = V, V_std
        
        #========== print ==========
        tend = time.time()
        print("state:", index[0], orb_state[index[0]], orbitals_array2str(orb_state[index[0]]))
        print(" "*8, "E: %.6f" % E, "(%.6f)" % E_std, 
                    " K: %.6f" % K, "(%.6f)" % K_std, 
                    " V: %.6f" % V, "(%.6f)" % V_std)
        print(" "*8, "ac: %.4f" % accept_rate, " dt: %.3f" % (tend - tstart), flush=True)

#========== print ==========

print("\n"+"="*80)
print("file_name:", file_name)
print("\nSummarize (1): E")
for jj in cal_n_orbital:
    index = jnp.array([jj])[0]
    state_arr = orb_state[index]
    state_str = "[" + "%d "*len(state_arr) % tuple(state_arr) + "]"
    print("level %.3d:" %index, " E: %.12f (%.12f)" % (list_E[jj], list_Estd[jj]),
                state_str, orbitals_array2str(orb_state[index])) 

# print("\nSummarize (2): E")
# for jj in cal_n_orbital:
#     index = jnp.array([jj])[0]
#     print("level %.3d:" %index, 
#             " E: %.6f (%.6f)" % (list_E[jj], jnp.sqrt(list_Estd[jj])),
#             orbitals_array2str(orb_state[index])) 

# print("\nSummarize (3): E - E0")
# for jj in cal_n_orbital:
#     index = jnp.array([jj])[0]
#     if jj == 0:
#         print("level %.3d:" %index,
#                 " E: %.2f (%.2f)" % (list_E[jj], list_Estd[jj]), 
#                 orbitals_array2str(orb_state[index])) 
#     else:
#         print("level %.3d:" %index, 
#                 " E: %.2f (%.2f)" % ((list_E[jj] - list_E[0]), 
#                                     (jnp.sqrt(list_Estd[jj]**2 + list_Estd[0]**2))),
#                 orbitals_array2str(orb_state[index])) 

print("")