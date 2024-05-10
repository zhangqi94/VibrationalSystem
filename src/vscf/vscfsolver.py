import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

import numpy as np
import time
import copy
from veff import Veff
from integrallist import get_psi_q_psi


def solve_vscf(n, w_indices, k2, k3, k4, state_index, Nlev=10, iters_vscf=10, print_level=1):

    psi_q0_psi, psi_q1_psi, psi_q2_psi, psi_q3_psi, psi_q4_psi, psi_qT_psi = get_psi_q_psi(Nlev)
    integral_table = [ psi_q1_psi, psi_q2_psi, psi_q3_psi]
    C = np.array([np.ones((Nlev, Nlev))/np.sqrt(Nlev) for _ in range(n)])
    mu1_vals, mu2_vals = np.meshgrid(range(Nlev), range(Nlev))
    
    # get effective hamiltonian
    def get_heff_func(modei, veff_obj):
        
        veff3_X2, veff3_X1, veff3_X0 = veff_obj.veff3(modei)
        veff4_X3, veff4_X2, veff4_X1, veff4_X0 = veff_obj.veff4(modei)
        
        def get_heff_matrix(mu1, mu2):
            temp_qT = psi_qT_psi[mu1, mu2] * w_indices[modei]
            temp_q0 = psi_q0_psi[mu1, mu2]                             * (veff3_X0 + veff4_X0)
            temp_q1 = psi_q1_psi[mu1, mu2] / (w_indices[modei]**(1/2)) * (veff3_X1 + veff4_X1)
            temp_q2 = psi_q2_psi[mu1, mu2] / (w_indices[modei]**(1))   * (veff3_X2 + veff4_X2 + k2[modei, modei])
            temp_q3 = psi_q3_psi[mu1, mu2] / (w_indices[modei]**(3/2)) * (veff4_X3 + k3[modei, modei, modei])
            temp_q4 = psi_q4_psi[mu1, mu2] / (w_indices[modei]**(2))   * (k4[modei, modei, modei, modei])
            heff_matrix = temp_qT + temp_q0 + temp_q1 + temp_q2 + temp_q3 + temp_q4
            return heff_matrix
        
        return get_heff_matrix

    ##########################################################################################
    print("solve vibrational self-consistent field (vscf)")
    
    E = np.zeros((n, Nlev))
    for iters in range(iters_vscf):
        t10 = time.time()
        
        for modei in range(n):
            t1 = time.time()
            
            C0 = copy.deepcopy(C)
            veff_obj = Veff(C, w_indices, k3, k4, integral_table, state_index)
            heff_matrix = get_heff_func(modei, veff_obj)(mu1_vals, mu2_vals)
            E_new, C_new = np.linalg.eigh(heff_matrix)
            
            C[modei, :] = C_new
            E[modei, :] = E_new
            
            sum_E = np.sum(E)
            norm_dC = np.linalg.norm((C - C0).reshape(-1))
            
            t2 = time.time()
            
            if print_level == 2:
                print("  iters: %2d,  mode: %2d,  sum_E:  %.6f,  norm_dC:  %.3e,  dt: %.3f" 
                        %(iters, modei, sum_E, norm_dC, t2-t1), flush=True)

        t20 = time.time()
        if print_level == 1:
            print("  iters: %2d,  sum_E:  %.6f,  norm_dC:  %.3e,  dt: %.3f" 
                    %(iters, sum_E, norm_dC, t20-t10), flush=True)
            
    return C, E
    
    
    