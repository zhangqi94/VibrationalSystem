import itertools 
import os
import numpy as np
os.environ["NPY_DEFAULT_FLOAT_TYPE"] = "float64"

class Vcoupled(object):

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
        
    def _get_CQ_sum_list(self):
        _CQ1_sum_list = [self._CQ1_sum(modej) for modej in range(self._M)]
        _CQ2_sum_list = [self._CQ2_sum(modej) for modej in range(self._M)]
        _CQ3_sum_list = [self._CQ3_sum(modej) for modej in range(self._M)]
        return np.array(_CQ1_sum_list), np.array(_CQ2_sum_list), np.array(_CQ3_sum_list)
       
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
    
    def vcoupled3(self):
        iter_list = list(itertools.combinations_with_replacement(range(self._M),r=3))
        vcoupled3_num = 0.0
        
        for i,j,k in iter_list:
            index = np.array([i, j, k], dtype=np.int32)
            q_order = [np.count_nonzero(index == i) for i in range(self._M)]
            tempX = self._force_cubic[i,j,k]
            for modej in range(self._M):
                if q_order[modej] == 1:
                    tempX *= self._CQ1_sum_list[modej]
                if q_order[modej] == 2:
                    tempX *= self._CQ2_sum_list[modej]
                if q_order[modej] == 3:
                    tempX *= 0.0
            vcoupled3_num += tempX
                
        return vcoupled3_num
         
    def vcoupled4(self):
        iter_list = list(itertools.combinations_with_replacement(range(self._M),r=4))
        vcoupled4_num = 0.0
        
        for i,j,k,l in iter_list:
            index = np.array([i, j, k, l], dtype=np.int32)
            q_order = [np.count_nonzero(index == i) for i in range(self._M)]
            tempX = self._force_quart[i,j,k,l]
            for modej in range(self._M):
                if q_order[modej] == 1:
                    tempX *= self._CQ1_sum_list[modej]
                if q_order[modej] == 2:
                    tempX *= self._CQ2_sum_list[modej]
                if q_order[modej] == 3:
                    tempX *= self._CQ3_sum_list[modej]
                if q_order[modej] == 4:
                    tempX *= 0.0
            vcoupled4_num += tempX
                
        return vcoupled4_num

####################################################################################

if __name__ == "__main__":
    1
        
####################################################################################
