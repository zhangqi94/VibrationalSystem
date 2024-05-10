import itertools 
import numpy as np

class Veff(object):
    """Supporting Veff calculation of J. Chem. Theory Comput. 2019, 15, 3766-3777
    Following eq(15)

    Attributes:
        self._coefficients: (N_{lev},N_{lev}) the combination coefficients
            in eq 10.
        self._frequencies: (M,M), here M is the total number of modes, 
            and self._frequencies is the main frequencies of the system.
        self._cubic_force: eta_t(cubic terms) in both eq 7 and eq 14
        self._quartic_force: eta_t(quartic terms) in both eq 7 and eq 14
        self._integral_table: integral list of different hermitian integral,
            <psi^mu|Q^p|psi^nu>, eq 19. NOTE: leading dimension corresponds
            to different type of integral table, for example, 
            integral_table = [psi_q1_psi, psi_q2_psi, psi_q3_psi, psi_q4_psi, psi_qT_psi]
        self._state_index: (M,) the list of \vec{n} = (n1, n2, n3, ... nM)
        self._M: the number of modes.
    """
    def __init__(
            self,
            coefficients: np.ndarray,
            frequencies: np.ndarray,
            cubic_force: np.ndarray,
            quartic_force: np.ndarray,
            integral_table: np.ndarray,
            state_index: np.ndarray,
            ) -> None:
        self._coefficients = coefficients
        self._frequencies = frequencies
        self._cubic_force = cubic_force
        self._quartic_force = quartic_force
        self._integral_table = integral_table
        self._state_index = state_index
        self._M = len(state_index)
        self._psi_q1_psi = integral_table[0]
        self._psi_q2_psi = integral_table[1]
        self._psi_q3_psi = integral_table[2]
        self._psi_q4_psi = integral_table[3]

    def _single_C_Q_sum(self, modei:int) -> float:
        """Perform sum_{nu,mu} C_{nu,nj} C_{mu,nj} <psi_j^{nu}|Q_j|psi_j^{mu}>"""
        return np.einsum("i,ij,j", 
                         self._coefficients[modei][:,self._state_index[modei]],
                         self._psi_q1_psi,
                         self._coefficients[modei][:,self._state_index[modei]]) / np.sqrt(self._frequencies[modei])
 
    def _double_C_Q_sum(self, modei:int) -> float:
        """Perform sum_{nu,mu} C_{nu,nj} C_{mu,nj} <psi_j^{nu}|Q_j^2|psi_j^{mu}>"""
        return np.einsum("i,ij,j", 
                         self._coefficients[modei][:,self._state_index[modei]],
                         self._psi_q2_psi,
                         self._coefficients[modei][:,self._state_index[modei]]) / self._frequencies[modei]
  
    def _triple_C_Q_sum(self, modei:int) -> float:
        """Perform sum_{nu,mu} C_{nu,nj} C_{mu,nj} <psi_j^{nu}|Q_j^3|psi_j^{mu}>"""
        return np.einsum("i,ij,j", 
                         self._coefficients[modei][:,self._state_index[modei]],
                         self._psi_q3_psi,
                         self._coefficients[modei][:,self._state_index[modei]]) / self._frequencies[modei]**(3/2)
   
    def _quardra_C_Q_sum(self, modei:int) -> float:
        """Perform sum_{nu,mu} C_{nu,nj} C_{mu,nj} <psi_j^{nu}|Q_j^4|psi_j^{mu}>"""
        return np.einsum("i,ij,j", 
                         self._coefficients[modei][:,self._state_index[modei]],
                         self._psi_q4_psi,
                         self._coefficients[modei][:,self._state_index[modei]]) / self._frequencies[modei]**2
        
    def v_eff_3(
            self,
            modei:int,
            qi_order:int,
            ) -> float:
        """Calculate the X_t in V_{eff,i}^n (Q_i) = sum_t X_t Q_i^{pti}
        Equation 15 of J. Chem. Theory Comput. 2019, 15, 3766-3777
        under constraint that sum_j p_{tj} == 3

        Args: 
            modei: the index of the chosen mode, aka, i in Q_i.
            qi_order: the order pri for the desired rth term. 
        Returns:
            X_t: the coefficient X_t in the summation sum_t X_t Q_i^{pti}
                given modei and qi_order. 
        """
        index_exclude_i = list(range(self._M))
        index_exclude_i.remove(modei)
        if qi_order == 2:
            X_list = []
            for i in index_exclude_i:
                X_list.append(
                    self._single_C_Q_sum(modei=i)
                    * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k
                        self._cubic_force[modei,modei,i]
                        +self._cubic_force[i,modei,modei]
                    )
                )
            X_t = np.sum(X_list)
            return X_t
        elif qi_order == 1:
            iter_list = list(itertools.combinations_with_replacement(index_exclude_i,r=2))
            X_list = []
            for i,j in iter_list:
                if i == j:
                    X_list.append(
                        self._double_C_Q_sum(modei=i)
                        * (# deal with non-symmetric force constants
                            # with constrainst i<=j<=k
                            self._cubic_force[modei,i,j]
                            +self._cubic_force[i,j,modei]
                        )
                    )
                else:
                    X_list.append(
                        self._single_C_Q_sum(modei=i) 
                        * self._single_C_Q_sum(modei=j)
                        * (# deal with non-symmetric force constants
                            # with constrainst i<=j<=k
                            self._cubic_force[modei,i,j]
                            +self._cubic_force[i,modei,j]
                            +self._cubic_force[i,j,modei]
                        )
                    )
            X_t = np.sum(X_list)
            return X_t
        elif qi_order == 0:
            iter_list = list(itertools.combinations_with_replacement(index_exclude_i,r=3))
            X_list = []
            for i,j,k in iter_list:
                if i != j and j != k and k != i:
                    X_list.append(
                        self._single_C_Q_sum(modei=i)
                        * self._single_C_Q_sum(modei=j)
                        * self._single_C_Q_sum(modei=k)
                        * self._cubic_force[i,j,k]
                    )
                elif i == j and i != k:
                    X_list.append(
                        self._double_C_Q_sum(modei=i)
                        * self._single_C_Q_sum(modei=k)
                        * self._cubic_force[i,j,k]
                    )
                elif i == k and i != j:
                    X_list.append(
                        self._double_C_Q_sum(modei=i)
                        * self._single_C_Q_sum(modei=j)
                        * self._cubic_force[i,j,k]
                    )
                elif j == k and i != j:
                    X_list.append(
                        self._double_C_Q_sum(modei=j)
                        * self._single_C_Q_sum(modei=i)
                        * self._cubic_force[i,j,k]
                    )
            X_t = np.sum(X_list)
            return X_t
         
    def v_eff_4(
            self,
            modei:int,
            qi_order:int,
            ) -> float:
        """Calculate the X_t in V_{eff,i}^n (Q_i) = sum_t X_t Q_i^{pti}
        Equation 15 of J. Chem. Theory Comput. 2019, 15, 3766-3777
        under constraint that sum_j p_{tj} == 4

        Args: 
            modei: the index of the chosen mode, aka, i in Q_i.
            qi_order: the order pri for the desired rth term. 
        Returns:
            X_t: the coefficient X_t in the summation sum_t X_t Q_i^{pti}
                given modei and qi_order. 
        """
        index_exclude_i = list(range(self._M))
        index_exclude_i.remove(modei)
        if qi_order == 3:
            X_list = []
            for i in index_exclude_i:
                X_list.append(
                    self._single_C_Q_sum(modei=i)
                    * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k<=l
                        self._quartic_force[modei,modei,modei,i]
                        +self._quartic_force[i,modei,modei,modei]
                    )
                )
            X_t = np.sum(X_list)
            return X_t
        elif qi_order == 2:
            iter_list = list(itertools.combinations_with_replacement(index_exclude_i,r=2))
            X_list = []
            for i,j in iter_list:
                if i == j:
                    X_list.append(
                        self._double_C_Q_sum(modei=i)
                        * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k<=l
                        self._quartic_force[modei,modei,i,j]
                        +self._quartic_force[i,j,modei,modei]
                        )
                    )
                else:
                    X_list.append(
                        self._single_C_Q_sum(modei=i) 
                        * self._single_C_Q_sum(modei=j)
                        * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k<=l
                        self._quartic_force[modei,modei,i,j]
                        +self._quartic_force[i,modei,modei,j]
                        +self._quartic_force[i,j,modei,modei]
                        )
                    )
            X_t = np.sum(X_list)
            return X_t
        elif qi_order == 1:
            iter_list = list(itertools.combinations_with_replacement(index_exclude_i,r=3))
            X_list = []
            for i,j,k in iter_list:
                if i != j and j != k and k != i:
                    X_list.append(
                        self._single_C_Q_sum(modei=i)
                        * self._single_C_Q_sum(modei=j)
                        * self._single_C_Q_sum(modei=k)
                        * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k<=l
                        self._quartic_force[modei,i,j,k]
                        +self._quartic_force[i,modei,j,k]
                        +self._quartic_force[i,j,modei,k]
                        +self._quartic_force[i,j,k,modei]
                        )
                    )
                elif i == j and i != k:
                    X_list.append(
                        self._double_C_Q_sum(modei=i)
                        * self._single_C_Q_sum(modei=k)
                        * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k<=l
                        self._quartic_force[modei,i,j,k]
                        +self._quartic_force[i,j,modei,k]
                        +self._quartic_force[i,j,k,modei]
                        )    
                    )
                # elif i == k and i != j:
                #     X_list.append(
                #         self._double_C_Q_sum(modei=i)
                #         * self._single_C_Q_sum(modei=j)
                #         * (# deal with non-symmetric force constants
                #         # with constrainst i<=j<=k<=l
                #         self._quartic_force[modei,i,j,k]
                #         +self._quartic_force[i,modei,j,k]
                #         +self._quartic_force[i,j,modei,k]
                #         +self._quartic_force[i,j,k,modei]
                #         )
                #     )
                elif j == k and i != j:
                    X_list.append(
                        self._double_C_Q_sum(modei=j)
                        * self._single_C_Q_sum(modei=i)
                        * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k<=l
                        self._quartic_force[modei,i,j,k]
                        +self._quartic_force[i,modei,j,k]
                        +self._quartic_force[i,j,k,modei]
                        )
                    )
                elif i == j and j == k:
                    X_list.append(
                        self._triple_C_Q_sum(modei=i)
                        * (# deal with non-symmetric force constants
                        # with constrainst i<=j<=k<=l
                        self._quartic_force[modei,i,j,k]
                        +self._quartic_force[i,j,k,modei]
                        )
                    )
            X_t = np.sum(X_list)
            return X_t
        elif qi_order == 0:
            iter_list = list(itertools.combinations_with_replacement(index_exclude_i,r=4))
            X_list = []
            for i,j,k,l in iter_list:
                index_list = list([i,j,k,l])
                pri_dict = {
                    "1": [],
                    "2": [],
                    "3": [],
                    "4": [],
                }
                if len(set(index_list)) == 4: # no same indices
                    X_list.append(
                        self._single_C_Q_sum(modei=i)
                        * self._single_C_Q_sum(modei=j)
                        * self._single_C_Q_sum(modei=k)
                        * self._single_C_Q_sum(modei=l)
                        * self._quartic_force[i,j,k,l]
                    )
                elif len(set(index_list)) == 3: # two indices are the same while else are different
                    for term in index_list:
                        order = index_list.count(term)
                        pri_dict[f"{order}"].append(term)
                    for key in pri_dict.keys():
                        pri_dict[key] = list(set(pri_dict[key]))
                    product = 1
                    for single_term in pri_dict["1"]:
                        product *= self._single_C_Q_sum(modei=single_term)
                    for double_term in pri_dict["2"]:
                        product *= self._double_C_Q_sum(modei=double_term)
                    product *= self._quartic_force[i,j,k,l]
                    X_list.append(product)
                elif len(set(index_list)) == 2: # three indices are the same or two pairs.
                    for term in index_list:
                        order = index_list.count(term)
                        pri_dict[f"{order}"].append(term)
                    for key in pri_dict.keys():
                        pri_dict[key] = list(set(pri_dict[key]))
                    product = 1
                    for single_term in pri_dict["1"]:
                        product *= self._single_C_Q_sum(modei=single_term)
                    for double_term in pri_dict["2"]:
                        product *= self._double_C_Q_sum(modei=double_term)
                    for triple_term in pri_dict["3"]:
                        product *= self._triple_C_Q_sum(modei=triple_term)
                    product *= self._quartic_force[i,j,k,l]
                    X_list.append(product)
            X_t = np.sum(X_list)
            return X_t  

####################################################################################

if __name__ == "__main__":
    import sys 
    sys.path.append("./src/")
    from integrallist import get_psi_q_psi
    Nlev=8
    _, psi_q1_psi, psi_q2_psi, psi_q3_psi, psi_q4_psi, psi_qT_psi = get_psi_q_psi(Nlev)
    n=6
    dim=1
    num_orb=10
    alpha = 1000.0
    from potential import get_potential_energy_H2CO
    potential_energy, w_indices, k2, k3, k4 = get_potential_energy_H2CO(alpha=alpha)
    C = np.array([np.eye(Nlev) for _ in range(n)])
    coefficients = C
    frequencies = w_indices
    cubic_force = k3
    quartic_force = k4
    integral_table = [
        psi_q1_psi[:Nlev,:Nlev],
        psi_q2_psi[:Nlev,:Nlev],
        psi_q3_psi[:Nlev,:Nlev],
        psi_q4_psi[:Nlev,:Nlev],
        psi_qT_psi[:Nlev,:Nlev]]
    state_index = np.array([1,2,0,0,2,0])

    veff_obj = Veff(
        coefficients=coefficients,
        frequencies=frequencies,
        cubic_force=cubic_force,
        quartic_force=quartic_force,
        integral_table=integral_table,
        state_index=state_index
    )

    for i in range(6):
        for qi_order in range(4):
            X_r = veff_obj.v_eff_3(modei=i, qi_order=qi_order)
            print(f"modei = {i} and qi_order = {qi_order}\n"
                f"Get X_r = {X_r}")

    for i in range(6):
        for qi_order in range(5):
            X_r = veff_obj.v_eff_4(modei=i, qi_order=qi_order)
            print(f"modei = {i} and qi_order = {qi_order}\n"
                f"Get X_r = {X_r}")
        
####################################################################################
