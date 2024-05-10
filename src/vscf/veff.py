import itertools 
import os
import numpy as np
os.environ["NPY_DEFAULT_FLOAT_TYPE"] = "float64"

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
            force_cubic: np.ndarray,
            force_quart: np.ndarray,
            integral_table: np.ndarray,
            state_index: np.ndarray,
            ) -> None:
        self._coefficients = coefficients
        self._frequencies = frequencies
        self._force_cubic = force_cubic
        self._force_quart = force_quart
        self._state_index = state_index
        self._M = len(state_index)
        self._psi_q1_psi = integral_table[0]
        self._psi_q2_psi = integral_table[1]
        self._psi_q3_psi = integral_table[2]
        self._CQ1_sum_list, self._CQ2_sum_list, self._CQ3_sum_list = self._get_CQ_sum_list()
        
    def _CQ1_sum(self, modej:int) -> float:
        """Perform sum_{nu,mu} C_{nu,nj} C_{mu,nj} <psi_j^{nu}|Q_j|psi_j^{mu}>"""
        return np.einsum("i,ij,j", 
                            self._coefficients[modej,:,self._state_index[modej]],
                            self._psi_q1_psi / self._frequencies[modej]**(1/2) ,
                            self._coefficients[modej,:,self._state_index[modej]])

    def _CQ2_sum(self, modej:int) -> float:
        """Perform sum_{nu,mu} C_{nu,nj} C_{mu,nj} <psi_j^{nu}|Q_j^2|psi_j^{mu}>"""
        return np.einsum("i,ij,j", 
                            self._coefficients[modej,:,self._state_index[modej]],
                            self._psi_q2_psi / self._frequencies[modej] ,
                            self._coefficients[modej,:,self._state_index[modej]])

    def _CQ3_sum(self, modej:int) -> float:
        """Perform sum_{nu,mu} C_{nu,nj} C_{mu,nj} <psi_j^{nu}|Q_j^3|psi_j^{mu}>"""
        return np.einsum("i,ij,j", 
                            self._coefficients[modej,:,self._state_index[modej]],
                            self._psi_q3_psi / self._frequencies[modej] **(3/2),
                            self._coefficients[modej,:,self._state_index[modej]])
    
    def _get_CQ_sum_list(self):
        _CQ1_sum_list = [self._CQ1_sum(modej) for modej in range(self._M)]
        _CQ2_sum_list = [self._CQ2_sum(modej) for modej in range(self._M)]
        _CQ3_sum_list = [self._CQ3_sum(modej) for modej in range(self._M)]
        return np.array(_CQ1_sum_list), np.array(_CQ2_sum_list), np.array(_CQ3_sum_list)
    
    def veff3(self, modei:int):
        
        iter_list = list(itertools.combinations_with_replacement(range(self._M),r=3))

        X2_list = 0.0
        X1_list = 0.0
        X0_list = 0.0
        
        for i,j,k in iter_list:
            index = np.array([i, j, k], dtype=np.int32)
            q_order = [np.count_nonzero(index == i) for i in range(self._M)]

            if q_order[modei] == 2:
                #print(index, q_order, "qi_order = 2")
                tempX = self._force_cubic[i,j,k]
                for modej in [x for x in range(self._M) if x != modei]:
                    if q_order[modej] == 1:
                        tempX *= self._CQ1_sum_list[modej]
                X2_list += tempX

            if q_order[modei] == 1:
                #print(index, q_order, "qi_order = 1")
                tempX = self._force_cubic[i,j,k]
                for modej in [x for x in range(self._M) if x != modei]:
                    if q_order[modej] == 1:
                        tempX *= self._CQ1_sum_list[modej]
                    if q_order[modej] == 2:
                        tempX *= self._CQ2_sum_list[modej]
                X1_list += tempX
            
            if q_order[modei] == 0:
                #print(index, q_order, "qi_order = 0")
                tempX = self._force_cubic[i,j,k]
                for modej in [x for x in range(self._M) if x != modei]:
                    if q_order[modej] == 1:
                        tempX *= self._CQ1_sum_list[modej]
                    if q_order[modej] == 2:
                        tempX *= self._CQ2_sum_list[modej]
                    if q_order[modej] == 3:
                        tempX *= 0.0
                X0_list += tempX
                
        return X2_list, X1_list, X0_list
         
    def veff4(self, modei:int):

        iter_list = list(itertools.combinations_with_replacement(range(self._M),r=4))

        X3_list = 0.0
        X2_list = 0.0
        X1_list = 0.0
        X0_list = 0.0
        
        for i,j,k,l in iter_list:
            index = np.array([i, j, k, l], dtype=np.int32)
            q_order = [np.count_nonzero(index == i) for i in range(self._M)]

            #if q_order[modei] == 4:
                #print(index, q_order, "Nothing to do")

            if q_order[modei] == 3:
                #print(index, q_order, "qi_order = 3")
                tempX = self._force_quart[i,j,k,l]
                for modej in [x for x in range(self._M) if x != modei]:
                    if q_order[modej] == 1:
                        tempX *= self._CQ1_sum_list[modej]
                X3_list += tempX

            if q_order[modei] == 2:
                #print(index, q_order, "qi_order = 2")
                tempX = self._force_quart[i,j,k,l]
                for modej in [x for x in range(self._M) if x != modei]:
                    if q_order[modej] == 1:
                        tempX *= self._CQ1_sum_list[modej]
                    if q_order[modej] == 2:
                        tempX *= self._CQ2_sum_list[modej]
                X2_list += tempX
            
            if q_order[modei] == 1:
                #print(index, q_order, "qi_order = 1")
                tempX = self._force_quart[i,j,k,l]
                for modej in [x for x in range(self._M) if x != modei]:
                    if q_order[modej] == 1:
                        tempX *= self._CQ1_sum_list[modej]
                    if q_order[modej] == 2:
                        tempX *= self._CQ2_sum_list[modej]
                    if q_order[modej] == 3:
                        tempX *= self._CQ3_sum_list[modej]
                X1_list += tempX
                
            if q_order[modei] == 0:
                #print(index, q_order, "qi_order = 0")
                tempX = self._force_quart[i,j,k,l]
                for modej in [x for x in range(self._M) if x != modei]:
                    if q_order[modej] == 1:
                        tempX *= self._CQ1_sum_list[modej]
                    if q_order[modej] == 2:
                        tempX *= self._CQ2_sum_list[modej]
                    if q_order[modej] == 3:
                        tempX *= self._CQ3_sum_list[modej]
                    if q_order[modej] == 4:
                        tempX *= 0.0
                X0_list += tempX
                
        return X3_list, X2_list, X1_list, X0_list

####################################################################################

if __name__ == "__main__":
    1
        
####################################################################################
