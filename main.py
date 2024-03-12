import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import os
import time
import src.VMC

from src.potential import get_potential_energy_User

####################################################################################
print("jax.__version__:", jax.__version__)
import subprocess
print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True).stdout, flush=True)
key = jax.random.PRNGKey(42)

####################################################################################
import argparse
parser = argparse.ArgumentParser(description="simulation for the harmonic oscillator")

### folder to save data.
parser.add_argument("--folder", default="/data/zhangqidata/Vibration/", help="the folder to save data")

### physical parameters.
parser.add_argument("--mode", type=str, default="H2O", help="mode type: H2O, H2CO, C2H4O, CH3CN, NDCO, TOY2, TOY3, User."
                    "Note: the option User is for user specified potential files.")
parser.add_argument("--n", type=int, default=15, help="total number of modes")
parser.add_argument("--dim", type=int, default=1, help="spatial dimension")

parser.add_argument("--num_orb", type=int, default=6, help="number of orbitals")
parser.add_argument("--choose_orb", type=int, nargs="+", default=None, help="choose orbital")
# parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")

### parameters only for toy models!!!
parser.add_argument("--w2_indices", type=float, nargs="+", default=None, help="w^2")
parser.add_argument("--lam", type=float, nargs="+", default=None, help="lambda")

### flow model: neural autoregressive flow & deep dense sigmoid flow parameters.
parser.add_argument("--flow_type", type=str, default="RNVP", help="NAF or RNVP")
parser.add_argument("--flow_depth", type=int, default=8, help="NAF/Real NVP: number of layers")
parser.add_argument("--mlp_width", type=int, default=16, help="NAF/Real NVP: width of the hidden layers")
parser.add_argument("--mlp_depth", type=int, default=2, help="NAF/Real NVP: depth of the hidden layers")
parser.add_argument("--dsf_width", type=int, default=8, help="DSF: width of the hidden layers")
parser.add_argument("--dsf_depth", type=int, default=2, help="DSF: number of layers")

### training parameters.
parser.add_argument("--batch", type=int, default=800, help="batch size (per single gradient accumulation step)")
parser.add_argument("--acc_steps", type=int, default=8, help="gradient accumulation steps")
parser.add_argument("--num_devices", type=int, default=1, help="number of GPU devices")
parser.add_argument("--epoch_finished", type=int, default=0, help="number of epochs already finished.")
parser.add_argument("--epoch", type=int, default=2000, help="final epoch")
parser.add_argument("--weight_in_sampling", type=str, default="Equal", 
                    help=("Each state's weight in sampling: Equal, Ground-half, Manual"))
parser.add_argument("--num_ground_total", type=int, default=1,
                    help=("The TOTAL number of ground states that would be calculated in sampling."))

### technical miscellaneous
parser.add_argument("--hutchinson", action='store_true', help="use Hutchinson's trick to compute the laplacian")

### optimizer parameters.
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (valid only for adam)")
parser.add_argument("--clip_factor", type=float, default=5.0, help="clip factor for gradient")

### mcmc
parser.add_argument("--mc_therm", type=int, default=20, help="MCMC thermalization steps")
parser.add_argument("--mc_steps", type=int, default=100, help="MCMC update steps")
parser.add_argument("--mc_stddev", type=float, default=0.5, 
                    help="standard deviation of the Gaussian proposal in MCMC update")

parser.add_argument("--ckpt_epochs", type=int, default=100, help="save checkpoint every ckpt_epochs")

# User specified potential's directory
parser.add_argument("--User_potential_dir", 
                    type=str,
                    default=None,
                    help="Only needed if --mode==User"
                    "Specifying the folder to which the user specified potential"
                    "files exists.")

#========== args.params -> params ==========
args = parser.parse_args()

folder, mode, n, dim = args.folder, args.mode, args.n, args.dim
w2_indices, lam = jnp.array(args.w2_indices), jnp.array(args.lam)
num_orb, choose_orb = args.num_orb, args.choose_orb
flow_type = args.flow_type
flow_depth, mlp_width, mlp_depth = args.flow_depth, args.mlp_width, args.mlp_depth
dsf_width, dsf_depth = args.dsf_width, args.dsf_depth
batch, acc_steps, num_devices, epoch = args.batch, args.acc_steps, args.num_devices, args.epoch
epoch_finished = args.epoch_finished
weight_in_sampling = args.weight_in_sampling
num_ground_total = args.num_ground_total
hutchinson, lr = args.hutchinson, args.lr
clip_factor = args.clip_factor
mc_therm, mc_steps, mc_stddev = args.mc_therm, args.mc_steps, args.mc_stddev
ckpt_epochs = args.ckpt_epochs
user_potential_dir = args.User_potential_dir

####################################################################################
print("\n========== Initialize Hamiltonian ==========")
print("Harmonic oscillator:\n    mode: %s, n: %d, dim: %d" % (mode, n, dim), flush=True)

# mode type: H2O, C2H4O, CH3CN, H2CO, NDCO, mode-2 (toy model), mode-3 (toy model), User
if mode == "H2O":
    alpha = 1000.0  #scaling factor
    from src.potential import get_potential_energy_water
    potential_energy, w_indices = get_potential_energy_water(alpha=alpha)
elif mode == "H2CO":
    alpha = 1000.0
    from src.potential import get_potential_energy_H2CO
    potential_energy, w_indices = get_potential_energy_H2CO(alpha=alpha)
elif mode == "C2H4O":
    alpha = 1000.0
    from src.potential import get_potential_energy_C2H4O
    potential_energy, w_indices = get_potential_energy_C2H4O(alpha=alpha)
elif mode == "NDCO": # N-dimensional coupled oscillator
    alpha = 1.0
    from src.potential import get_potential_energy_NDCO, calculate_exact_energy_NDCO
    potential_energy, w_indices = get_potential_energy_NDCO(D=n)
elif mode == "TOY2": # two-mode system
    alpha = 1.0
    from src.potential import get_potential_energy_mode2
    potential_energy, w_indices = get_potential_energy_mode2(lam, w2_indices)
elif mode == "TOY3": # three-mode system
    alpha = 1.0
    from src.potential import get_potential_energy_mode3
    potential_energy, w_indices = get_potential_energy_mode3(lam, w2_indices)    
elif mode == "CH3CN":
    alpha = 1000.0
    from src.potential import get_potential_energy_CH3CN
    potential_energy, w_indices = get_potential_energy_CH3CN(alpha=alpha)
elif mode == "User":
    # User specified potential
    if user_potential_dir == None:
        raise ValueError(f"For --mode==User, a user specified potential"
                         f"must be provided through --User_potential_dir=path-to-directory!"
                          f" Currently get {user_potential_dir}")
    elif not os.path.isdir(user_potential_dir):
        raise FileNotFoundError(f"The given directory not found!"
                                f"Expected {user_potential_dir}")
    else:
        alpha = 1000.0
        potential_energy, w_indices = get_potential_energy_User(
            user_potential_file_dir=user_potential_dir,
            modes_num=n,
            alpha=alpha,
        )
else:
    # Undefines mode, raise an error
    raise ValueError(f"Undifined behaviour in --mode!\n"
                     f"For instruction, type 'python3 main.py --help'. \n Currently get {mode}")

if len(w_indices) != n: raise ValueError("Number of modes is not consistent!")
invsqrtw = 1/jnp.sqrt(w_indices)

####################################################################################
print("\n========== Initialize orbitals ==========")
from src.potential.orbitals import get_orbitals_1d, get_orbitals_indices_first, \
                                    orbitals_array2str, choose_orbitals
if dim != 1: raise ValueError("Only dim = 1 is supported!")
sp_orbitals, _ = get_orbitals_1d()

print("Total number of orbitals:", num_orb)
if mode == "NDCO": # calculate exact energy of NDCO
    orb_index, orb_state, orb_Es, nu = calculate_exact_energy_NDCO(D=n, num_orb=num_orb)
    print("nu:", nu, "\nExact energy of each energy level:")
    for ii in range(num_orb): 
        print("    %d, E: %.12f, level:" %(ii, orb_Es[ii]), orbitals_array2str(orb_state[ii]), flush=True)  
        
elif mode == "TOY2" or mode == "TOY3": # toy model: mode2 or mode3 
    if choose_orb == None:
        orb_index, orb_state, orb_Es = get_orbitals_indices_first(w_indices, max_orb=1000, num_orb = num_orb)
        print("Energy of non-interacting system:")
        for ii in range(num_orb):
            print("    %d, E: %.12f, level:" %(ii, orb_Es[ii]), orb_state[ii], flush=True)
    elif choose_orb != None:
        if len(choose_orb) != num_orb: raise ValueError("Number of orbitals is not consistent!")
        _, orb_state, orb_Es = get_orbitals_indices_first(w_indices, max_orb=1000, num_orb = 1000)
        orb_index, orb_state, orb_Es = choose_orbitals(orb_state, orb_Es * alpha, choose_orb)
        print("\nEnergy of non-interacting system:(choose states!)")
        for ii in range(num_orb):  
            print("    %d, E: %.2f, level:" %(ii, orb_Es[ii]), orbitals_array2str(orb_state[ii]), flush=True)
            
else: ## H2O, C2H4O, CH3CN, H2CO, User
    if choose_orb == None:
        orb_index, orb_state, orb_Es = get_orbitals_indices_first(w_indices, max_orb=1000, num_orb = num_orb)
        orb_Es = orb_Es * alpha
        print("Energy of non-interacting system:")
        for ii in range(num_orb):  
            print("    %d, E: %.2f, level:" %(ii, orb_Es[ii]), orbitals_array2str(orb_state[ii]), flush=True)
    elif choose_orb != None:
        if len(choose_orb) != num_orb: raise ValueError("Number of orbitals is not consistent!")
        _, orb_state, orb_Es = get_orbitals_indices_first(w_indices, max_orb=1000, num_orb = 1000)
        orb_index, orb_state, orb_Es = choose_orbitals(orb_state, orb_Es * alpha, choose_orb)
        print("\nEnergy of non-interacting system:(choose states!)")
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

params_flow = flow.init(key, jnp.zeros((n, dim)))
raveled_params_flow, _ = ravel_pytree(params_flow)
print("    parameters in the flow model: %d" % raveled_params_flow.size, flush=True)

#========== logpsi = logphi + 0.5*logjacdet ==========
import src.logpsi
logpsi_novmap = src.logpsi.make_logpsi(flow, sp_orbitals, orb_state, w_indices)
logphi, logjacdet = src.logpsi.make_logphi_logjacdet(flow, sp_orbitals, orb_state, w_indices)
logp = src.logpsi.make_logp(logpsi_novmap)

####################################################################################
print("\n========== Initialize optimizer ==========")
import optax
import src.sr

optimizer = optax.adam(lr)
print("Optimizer adam:\n    learning rate: %g" % lr)

####################################################################################
print("\n========== Check point ==========")
from src.utils import replicate, shard
import src.checkpoint 
from src.sample import init_state_indices

#========== create file path ==========
if mode in ["TOY2", "TOY3"]:
    mode_str = ("TOY_n_%d" % n) \
        + ("_ws" + "_%g"*n) % tuple(w2_indices) + ("_lam" + "_%g"*len(lam)) % tuple(lam)
else: # for H2O, C2H4O, H2CO, NDCO, CH3CN, User model
    mode_str = mode + ("_n_%d" % n)
    
if flow_type == "NAF":
    flow_str = ("_naf_%d" % flow_depth) + ("_mlp_%d_%d" % (mlp_width, mlp_depth)) \
        +  ("_dsf_%d_%d" % (dsf_width, dsf_depth))
elif flow_type == "RNVP":
    flow_str = ("_rnvp_%d" % flow_depth) + ("_mlp_%d_%d" % (mlp_width, mlp_depth))

if weight_in_sampling == "Equal": weightstr = "_wgt_eq"
elif weight_in_sampling == "Ground-half": weightstr = "_wgt_gh"
elif weight_in_sampling == "Manual": weightstr = "_wgt_ma" + ("_ngs_%d" % num_ground_total)

path = folder + mode_str + ("_alp_%g" % (alpha)) \
        + ("_cho" if choose_orb!=None else "") + ("_orb_%d" % num_orb) \
        + weightstr + flow_str \
        + ("_mcthr_%d_stp_%d_std_%g" % (mc_therm, mc_steps, mc_stddev)) \
        + ("_hut" if hutchinson else "") \
        + ("_adamlr_%g" % lr) + ("_clp_%g" % clip_factor) \
        + ("_bth_%d_acc_%d_nde_%d" % (batch, acc_steps, num_devices))

print("#file path:", path)
if not os.path.isdir(path):
    os.makedirs(path)
    print("#create path: %s" % path)

# Loading target checkpoint file
load_ckpt_filename = src.checkpoint.ckpt_filename(epoch_finished,path)

#========== initialize ==========
print("Number of GPU devices:", num_devices)
if num_devices != jax.device_count():
    raise ValueError("Expected %d GPU devices. Got %d." % (num_devices, jax.device_count()))

if batch % num_devices != 0:
    raise ValueError("Batch size must be divisible by the number of GPU devices. "
                        "Got batch = %d for %d devices now." % (batch, num_devices))
batch_per_device = batch // num_devices

if os.path.isfile(load_ckpt_filename):
    print(f"Load checkpoint file:{load_ckpt_filename}")
    ckpt = src.checkpoint.load_data(load_ckpt_filename)
    (keys, x, opt_state, params_flow,
     mode, w2_indices, lam,
     n, dim, num_orb, choose_orb,
     alpha, orb_index, orb_state, orb_Es,
     flow_type, flow_depth,
     mlp_width, mlp_depth,
     dsf_width, dsf_depth,
     batch, acc_steps, num_devices,
     hutchinson, lr, clip_factor,
     mc_therm, mc_steps, mc_stddev) = (
         ckpt["keys"], ckpt["x"], ckpt["opt_state"], ckpt["params_flow"],
         ckpt["mode"], ckpt["w2_indices"], ckpt["lam"],
         ckpt["n"], ckpt["dim"], ckpt["num_orb"], ckpt["choose_orb"],
         ckpt["alpha"], ckpt["orb_index"], ckpt["orb_state"], ckpt["orb_Es"],
         ckpt["flow_type"], ckpt["flow_depth"],
         ckpt["mlp_width"], ckpt["mlp_depth"],
         ckpt["dsf_width"], ckpt["dsf_depth"],
         ckpt["batch"], ckpt["acc_steps"], ckpt["num_devices"],
         ckpt["hutchinson"], ckpt["lr"], ckpt["clip_factor"],
         ckpt["mc_therm"], ckpt["mc_steps"], ckpt["mc_stddev"],
     )
    x, keys = shard(x), shard(keys)
    params_flow = replicate(params_flow,num_devices)

    #============= sampler ============
    state_indices, real_num_orb_in_state_indices = init_state_indices(
        orb_index=orb_index,
        num_orb=num_orb,
        num_devices=num_devices,
        weight_in_sampling=weight_in_sampling,
        batch_per_device=batch_per_device,
        num_ground_total=num_ground_total,
    )

else:
    print("No checkpoint file found. Start from scratch.")

    #========== sampler ==========
    state_indices, real_num_orb_in_state_indices = init_state_indices(
        orb_index=orb_index,
        num_orb=num_orb,
        num_devices=num_devices,
        weight_in_sampling=weight_in_sampling,
        batch_per_device=batch_per_device,
        num_ground_total=num_ground_total,
    )
    print("Initialize key and coordinate samples...", flush=True)
    x = jax.random.normal(key, (num_devices, batch_per_device*real_num_orb_in_state_indices, n, dim))
    keys = jax.random.split(key, num_devices)
    x, keys = shard(x), shard(keys)
    print("x.shape:", x.shape)

    #========== optimizer ==========
    print("initialize optimizer...", flush=True)
    opt_state = optimizer.init(params_flow)
    params_flow = replicate(params_flow, num_devices)

    #========== thermalize ==========
    for ii in range(mc_therm):
        t1 = time.time()
        keys, x, accept_rate = src.VMC.sample_x_mcmc(keys, state_indices, 
                                        logp, x, params_flow, mc_steps, mc_stddev, invsqrtw)
        t2 = time.time()
        print("---- thermal step: %d,  ac: %.4f,  dx: %.4f,  dt: %.3f ----" 
            % (ii+1, accept_rate[0], mc_stddev, t2-t1), flush=True)
        if accept_rate[0] > 0.525: mc_stddev *= 1.1
        if accept_rate[0] < 0.475: mc_stddev *= 0.9

#========== grad, laplacian ==========
logpsi, logpsi_grad_laplacian = src.logpsi.make_logpsi_grad_laplacian(logpsi_novmap, forloop=True,
                                        hutchinson=hutchinson, logphi=logphi, logjacdet=logjacdet)

#========== observable, loss function ==========
observable_and_lossfn = src.VMC.make_loss(logpsi, logpsi_grad_laplacian, potential_energy, clip_factor, 
                                          batch_per_device, real_num_orb_in_state_indices)

#========== update function ==========
from functools import partial
@partial(jax.pmap, axis_name="p",
        in_axes =(0, None, 0, 0, 0, 0, 0, 0, None),
        out_axes=(0, None, 0, 0, 0),
        static_broadcasted_argnums=8,
        donate_argnums=(2, 3))
def update(params_flow, opt_state, state_indices, x, key, 
           datas_acc, grads_acc, quant_score_acc, final_step):
    
    datas, quant_lossfn = observable_and_lossfn(params_flow, state_indices, x, key)

    grad_params_flow, quant_score = jax.jacrev(quant_lossfn)(params_flow)
    grads = grad_params_flow
    grads, quant_score = jax.lax.pmean((grads, quant_score), axis_name="p")
    datas_acc, grads_acc, quant_score_acc = jax.tree_map(lambda acc, i: acc + i, 
                            (datas_acc, grads_acc, quant_score_acc), 
                            (datas,     grads,     quant_score))

    if final_step:
        datas_acc, grads_acc, quant_score_acc = jax.tree_map(lambda acc: acc / acc_steps, 
                                                (datas_acc, grads_acc, quant_score_acc))
        grad_params_flow = grads_acc
        grad_params_flow = jax.tree_map(lambda grad, quant_score: grad - datas_acc["E_mean"] * quant_score,
                                            grad_params_flow, quant_score_acc)
        grads_acc = grad_params_flow
            
        updates, opt_state = optimizer.update(grads_acc, opt_state)
        params_flow = optax.apply_updates(params_flow, updates)

    return params_flow, opt_state, datas_acc, grads_acc, quant_score_acc

#========== open file ==========
log_filename = os.path.join(path, "data.txt")
print("#data name: ", log_filename, flush=True)
f = open(log_filename, "w" if epoch_finished == 0 else "a",
            buffering=1, newline="\n")
####################################################################################
print("\n========== Training ==========")

#========== circulate ==========
t0 = time.time()
for ii in range(epoch_finished+1, epoch+1):
    t1 = time.time()
    
    datas_acc = replicate({"E_mean": 0., "E2_mean": 0., 
                           "K_mean": 0., "K2_mean": 0., 
                           "V_mean": 0., "V2_mean": 0.,
                           "LE_mean": 0., "LE2_mean": 0.,
                           "LK_mean": 0., "LK2_mean": 0.,
                           "LV_mean": 0., "LV2_mean": 0.
                           }, num_devices)
    grads_acc = shard( jax.tree_map(jnp.zeros_like, params_flow))
    quant_score_acc = shard(jax.tree_map(jnp.zeros_like, params_flow) )
    accept_rate_acc = shard(jnp.zeros(num_devices))

    for acc in range(acc_steps):   
        keys, x, accept_rate = src.VMC.sample_x_mcmc(keys, state_indices, 
                                        logp, x, params_flow, mc_steps, mc_stddev, invsqrtw)

        accept_rate_acc += accept_rate
        final_step = (acc == (acc_steps-1))
        
        params_flow, opt_state, datas_acc, grads_acc, quant_score_acc \
                        = update(params_flow, opt_state, state_indices, x, keys, 
                                 datas_acc, grads_acc, quant_score_acc, final_step)
        
    data = jax.tree_map(lambda x: x[0], datas_acc)
    accept_rate = accept_rate_acc[0] / acc_steps
    LE, LE2_mean = data["LE_mean"]*alpha, data["LE2_mean"]*(alpha**2)
    LK, LK2_mean = data["LK_mean"]*alpha, data["LK2_mean"]*(alpha**2)
    LV, LV2_mean = data["LV_mean"]*alpha, data["LV2_mean"]*(alpha**2)
    LE_std = jnp.sqrt((LE2_mean - LE**2) / (batch*acc_steps))
    LK_std = jnp.sqrt((LK2_mean - LK**2) / (batch*acc_steps))
    LV_std = jnp.sqrt((LV2_mean - LV**2) / (batch*acc_steps))
    
    #========== print ==========
    t2 = time.time()
    
    if mode in ["NDCO", "TOY2", "TOY3"]:
        print("iter: %05d" % ii,
            " E: %.6f (%.6f)  K: %.6f (%.6f)  V: %.6f (%.6f)"
            % (LE, LE_std, LK, LK_std, LV, LV_std),
            " ac: %.4f  dx: %.4f  dt: %.3f" % (accept_rate, mc_stddev, t2-t1), flush=True)
    else:
        print("iter: %05d" % ii,
            " E: %.2f (%.2f)  K: %.2f (%.2f)  V: %.2f (%.2f)"
            % (LE, LE_std, LK, LK_std, LV, LV_std),
            " ac: %.4f  dx: %.4f  dt: %.3f" % (accept_rate, mc_stddev, t2-t1), flush=True)

    #========== save ==========
    f.write( ("%6d" + "  %.12f"*6 + "  %.9f"*3 + "\n") 
            % (ii, LE, LE_std, LK, LK_std, LV, LV_std, accept_rate, mc_stddev, t2-t1))

    if ii % ckpt_epochs == 0:
        ckpt = {"keys": keys, "x": x, "opt_state": opt_state,
                "params_flow": jax.tree_map(lambda x: x[0], params_flow), 
                "mode": mode, "w2_indices": w2_indices, "lam": lam,
                "n": n, "dim": dim, "num_orb": num_orb, "choose_orb": choose_orb,
                "alpha": alpha,
                "orb_index": orb_index, "orb_state": orb_state, "orb_Es": orb_Es,
                "flow_type": flow_type,
                "flow_depth": flow_depth, "mlp_width": mlp_width, "mlp_depth": mlp_depth,
                "dsf_width": dsf_width, "dsf_depth": dsf_depth,
                "batch": batch, "acc_steps": acc_steps, "num_devices": num_devices,
                "hutchinson": hutchinson, "lr": lr, "clip_factor": clip_factor,
                "mc_therm": mc_therm, "mc_steps": mc_steps, "mc_stddev": mc_stddev,
                }
        save_ckpt_filename = src.checkpoint.ckpt_filename(ii, path)
        src.checkpoint.save_data(ckpt, save_ckpt_filename)
        print("save file: %s" % save_ckpt_filename, flush=True)
        print("total time used: %.3fs (%.3fh),  training speed: %.3f epochs per hour." 
              % ((t2-t0), (t2-t0)/3600, 3600.0/(t2-t0)*ii), flush=True)

    if accept_rate > 0.525: mc_stddev *= 1.1
    if accept_rate < 0.475: mc_stddev *= 0.9

f.close()
