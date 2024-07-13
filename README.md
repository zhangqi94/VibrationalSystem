# Solving vibrational Hamiltonians with neural canonical transformation

This project is designed for the study of vibrational systems characterized by their potential energy surfaces (PES).


## Model
When analyzing vibrational systems, we characterize them by their vibrational degrees of freedom, denoted as $D$, within the normal coordinates $\bm{x}=(x_1,x_2,...,x_D)\in \mathcal{R}^D$. The system's Hamiltonian is described by:
$$
H = - \frac{1}{2} \sum_{i=1}^D  \frac{\partial^2}{\partial x_i^2} 
+ V(\bm{x}).
$$
where each $x_i$ represents the coordinate associated with a distinct vibrational mode, and $V(\bm{x})$ signifies the potential energy surface (PES). The PES can take various forms, including series expansions such as:
$$
	V(\bm{x}) = \sum_{i=1}^D \frac{1}{2} \omega_i^2 x_i^2
    + \sum_{i,j,k}^D \frac{1}{6} \phi_{ijk} x_i x_j x_k
    + \sum_{i,j,k,l}^D \frac{1}{24} \phi_{ijkl} x_i x_j x_k x_l.
$$

## Running the program

### Requirements
A GPU is highly recommended for optimal performance. 
Ensure your environment meets the following specifications:
- Python >= 3.11
- jax >= 0.4.24
- dm-haiku >= 0.0.11
- optax >= 0.1.9 


### Training flow model and get energy of eigenstates

Run `python main.py --help` to check out the avaiable parameters for the code of training flow model.

A basic example of running the main program:
```
python3 ./main.py  \
    --folder "./C2H4O/"  \
    --mode "CH3CN"  --n=12  --num_orb=100  --weight_in_sampling "Equal"  \
    --flow_type "RNVP"  --flow_depth=16  --mlp_width=64  --mlp_depth=3 \
    --batch=1024  --acc_steps=1   --num_devices=1  --epoch=50000  \
    --clip_factor=5  --lr=1e-3  --mc_therm=20  --mc_steps=200  --mc_stddev=0.2 
```

To calculate the energy of each state after the model has been trained:
```
python3 ./main_getenergy.py  \
    --file_name "xxx/epoch_010000.pkl"
    --num_devices=1  --batch=65536  --acc_steps=8  --mc_therm=20  --mc_steps=500  --mc_stddev=0.2
```

### Easy-to-use api for other potential energy surfaces

We provide easy-to-use api to make user-specified calculations.

You can place with your own potential energy under the folder `user_api/` and follow the following steps to make vibrational calculation of your own potential!

1. Place your potential energy files, including `w0.txt`, `cubic.txt` and `quartic.txt` under `user_api/Your_System/`.
   - Here `Your_System` refers to the system's name.
   - An example is provided under `user_api/User_potential_1`, containing potential energies of C2H4O, for demonstration purposes.
2. Invoke the scripts as previously described, making changes to `--mode=User` and `--n=modes_number` corresponding to your system, and additionally provide  `--User_potential_dir="./user_api/User_potential_1/"`.
   - or example, the following commands would invoke a ground state calculation with user-provided potential energies:
   ```bash
    python3 ./main.py  \
    --folder "/{Path-To-Data}/User_potential_1/" \
    --mode "User"  --n=15  --num_orb=1 \
    --weight_in_sampling "Equal"  \
    --flow_type "RNVP"  --flow_depth=16  --mlp_width=128  --mlp_depth=2 \
    --batch=65536  --acc_steps=1   --num_devices=1  --epoch=20000  \
    --clip_factor=5  --lr=1e-3  \
    --mc_therm=20  --mc_steps=200  --mc_stddev=0.2 \
    --User_potential_dir="./user_api/User_potential_1/"
   ```
   - Replace `{Path-To-Data}` with the directory that you would like to save data.
- Note here that the potential energy files are denoted as follow:
  - For w0.txt, `i i w0_i`: the quadratic force constants (harmonic frequencies/2).
  - For cubic.txt, `i j k k30_{i,j,k}`: the cubic force constants.
  - For quartic.txt, `i j k l k_40_{i,j,k,l}`: the quartic force constants.

### Invocation Arguments
The arguments used to run the program are highly tunable.
Key arguments in `main.py` include:

- `--folder`: Destination folder for training data.
- `--mode`: Model selection. Available: `H2O`, `H2CO`, `C2H4O`, `CH3CN` are real molecules, `NDCO` for N dimensional coupled oscillator, `TOY2` and `TOY3` for 2 modes and 3 modes toy model, and `User` for user specified model.
- `--n`: Total vibrational modes to compute.
- `--num_orb`: Number of orbitals for calculation.
Typically, set `--num_orb=1` invokes a ground state calculation.
If `--num_orb=m` then the lowest m excitation states
would be calculated. Note: this parameter would left 
unchanged regardless of the choice of
 `--weight_in_sampling`, meaning even if 
`--weight_in_sampling=Ground_half`, the ground state
is only counted once in `--num_orb=m`.
- `--choose_orb`: Specify desired orbitals **(must maintain default in main function)**.
- `--flow_type`: Type of flow: RNVP or NAF. It is recommended to choose RNVP.
- `--flow_depth`: Number of layers for flow models.
- `--mlp_width`: Width of hidden layers for flow models.
- `--mlp_depth`: Depth of hidden layers for flow models.
- `--batch`: Batch size per gradient accumulation step.
- `--acc_steps`: Gradient accumulation steps.
- `--num_devices`: Total GPUs for execution.
- `--epoch_finished`: Finished training epochs for loading checkpoint.
- `--epoch`: Total epochs for training.
- `--weight_in_sampling`: State weight in sampling: Equal, Ground-half, Manual.
Equal for all states sampling with same weight,
Ground-half for manually setting ground state to have weight=1/2 for all the states, 
Manual for manually setting the total number of ground states that would 
appear in the real orbital index. NOTE: if set this argument to manual, then `--num_ground_total` must be 
specified.
- `--num_ground_total`: Total ground states for sampling (if `weight_in_sampling==Manual`)
- `--lr`: learning rate (valid only for adam).
- `--clip_factor`: clip factor for gradient.
- `--mc_therm`: MCMC thermalization steps.
- `--mc_steps`: MCMC update steps.
- `--mc_stddev`: standard deviation of the Gaussian 
proposal in MCMC update.
- `--User_potential_dir`: User-specified potential directory (if `--mode==User`).
- For default settings, refer to `python3 main.py --help`.



The arguments in `main_getenergy.py` is highly 
identical to that of `main.py`. The only thing 
one needs to keep in mind is that here `--choose_orb` 
plays a more pragmatic role, designating the
orbitals to be chosen to calculate energies.

## To cite

```
@article{10.1063/5.0209255,
    author = {Zhang, Qi and Wang, Rui-Si and Wang, Lei},
    title = "{Neural canonical transformations for vibrational spectra of molecules}",
    journal = {The Journal of Chemical Physics},
    volume = {161},
    number = {2},
    pages = {024103},
    year = {2024},
    month = {07},
    issn = {0021-9606},
    doi = {10.1063/5.0209255},
    url = {https://doi.org/10.1063/5.0209255},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0209255/20033705/024103\_1\_5.0209255.pdf},
}
```

## Training data

The corresponding data is available at [VibrationalSystemData](https://github.com/zhangqi94/VibrationalSystemData)
