import unittest
import sys 
sys.path.append("./src/vscf/")
import numpy as np

from veff import Veff
from integrallist import get_psi_q_psi
 
class TestVeff_M_2(unittest.TestCase):
    """Test Veff."""
 
    def setUp(self):
        "Test Veff for M=2, Nlev=2, state_index=[0 0]"

        Nlev=2
        n=2

        psi_q0_psi, psi_q1_psi, psi_q2_psi, psi_q3_psi, psi_q4_psi, psi_qT_psi = get_psi_q_psi(Nlevels=Nlev)
    
        self.einsum_vector = np.ones(Nlev)
        C = np.array([np.ones((Nlev, Nlev)) for _ in range(n)])
        self.coefficients = C
        self.frequencies = np.ones(n)
        self.cubic_force = np.ones((n,n,n))
        self.quartic_force = np.ones((n,n,n,n))
        self.state_index = np.zeros_like(list(range(n)))
        self.integral_table = [
            psi_q1_psi,
            psi_q2_psi,
            psi_q3_psi,
            psi_q4_psi,
            psi_qT_psi,
        ]
        self.veff_object = Veff(
            coefficients=self.coefficients,
            frequencies=self.frequencies,
            cubic_force=self.cubic_force,
            quartic_force=self.quartic_force,
            integral_table=self.integral_table,
            state_index=self.state_index
        )
        self.result_3 = {
            "modei=0,qi_order=1":2*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector),
            "modei=0,qi_order=2":2*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector),
            "modei=1,qi_order=1":2*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector),
            "modei=1,qi_order=2":2*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector),
        }
        self.result_4 = {
            "modei=0,qi_order=1":2*np.einsum("i,ij,j",self.einsum_vector,psi_q3_psi,self.einsum_vector),
            "modei=0,qi_order=2":2*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector),
            "modei=0,qi_order=3":2*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector),
            "modei=1,qi_order=1":2*np.einsum("i,ij,j",self.einsum_vector,psi_q3_psi,self.einsum_vector),
            "modei=1,qi_order=2":2*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector),
            "modei=1,qi_order=3":2*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector),
        }
 
    def tearDown(self):
        pass
 
    def test_v_eff_3(self):
        """Test case for v_eff_3."""
        modei_list = list(range(2))
        qi_order_list = list(range(1,3))
        
        for modei in modei_list:
            for qi_order in qi_order_list:
                # print(f"modei={modei}\nqi_order={qi_order}")
                result = self.result_3[f"modei={modei},qi_order={qi_order}"]
                calculated = self.veff_object.v_eff_3(
                    modei=modei,
                    qi_order=qi_order,
                )
                np.testing.assert_allclose(
                    result,
                    calculated,
                )
  
    def test_v_eff_4(self):
        """Test case for v_eff_4."""
        modei_list = list(range(2))
        qi_order_list = list(range(1,4))
        
        for modei in modei_list:
            for qi_order in qi_order_list:
                result = self.result_4[f"modei={modei},qi_order={qi_order}"]
                calculated = self.veff_object.v_eff_4(
                    modei=modei,
                    qi_order=qi_order,
                )
                np.testing.assert_allclose(
                    result,
                    calculated,
                )
 
class TestVeff_M_4(unittest.TestCase):
    """Test Veff."""
 
    def setUp(self):
        "Test Veff for M=4, Nlev=2, state_index=[0 0 0 0]"

        Nlev=2
        n=4

        psi_q0_psi, psi_q1_psi, psi_q2_psi, psi_q3_psi, psi_q4_psi, psi_qT_psi = get_psi_q_psi(Nlevels=Nlev)
   
        self.einsum_vector = np.ones(Nlev)
        C = np.array([np.ones((Nlev, Nlev)) for _ in range(n)])
        self.n = n
        self.Nlev = Nlev
        self.coefficients = C
        self.frequencies = np.ones(n)
        self.cubic_force = np.ones((n,n,n))
        self.quartic_force = np.ones((n,n,n,n))
        self.state_index = np.zeros_like(list(range(n)))
        self.integral_table = [
            psi_q1_psi,
            psi_q2_psi,
            psi_q3_psi,
            psi_q4_psi,
            psi_qT_psi,
        ]
        self.veff_object = Veff(
            coefficients=self.coefficients,
            frequencies=self.frequencies,
            cubic_force=self.cubic_force,
            quartic_force=self.quartic_force,
            integral_table=self.integral_table,
            state_index=self.state_index
        )
        self.result_3 = {
            "modei=0,qi_order=0": (
                6*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector)*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)
                + np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)**3
            ),
            "modei=0,qi_order=1":(
                2*3*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector)
                + 3*3*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)**2
            ),
            "modei=0,qi_order=2":(
                2*3*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)
            ),
        }
        self.result_4 = {
            "modei=0,qi_order=0":(
                6*np.einsum("i,ij,j",self.einsum_vector,psi_q3_psi,self.einsum_vector)*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)
                + 3*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector)**2
                + 3*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector)*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)**2
            ),
            "modei=0,qi_order=1":(
                2*3*np.einsum("i,ij,j",self.einsum_vector,psi_q3_psi,self.einsum_vector)
                + 3*6*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector)*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)
                + 4*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)**3
            ),
            "modei=0,qi_order=2":(
                2*3*np.einsum("i,ij,j",self.einsum_vector,psi_q2_psi,self.einsum_vector)
                + 3*3*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)**2
            ),
            "modei=0,qi_order=3":(
                2*3*np.einsum("i,ij,j",self.einsum_vector,psi_q1_psi,self.einsum_vector)
            ),
        }
 
    def tearDown(self):
        pass
 
    def test_v_eff_3(self):
        """Test case for v_eff_3."""
        modei = 0
        qi_order_list = list(range(3))
        
        for qi_order in qi_order_list:
            # print(f"modei={modei}\nqi_order={qi_order}")
            result = self.result_3[f"modei={modei},qi_order={qi_order}"]
            calculated = self.veff_object.v_eff_3(
                modei=modei,
                qi_order=qi_order,
            )
            np.testing.assert_allclose(
                result,
                calculated,
            )
  
    def test_v_eff_4(self):
        """Test case for v_eff_4."""
        modei = 0
        qi_order_list = list(range(4))
        
        for qi_order in qi_order_list:
            print(f"Testing qi_order={qi_order}")
            result = self.result_4[f"modei={modei},qi_order={qi_order}"]
            print(f"result={result}")
            calculated = self.veff_object.v_eff_4(
                modei=modei,
                qi_order=qi_order,
            )
            np.testing.assert_allclose(
                result,
                calculated,
            )

if __name__ == '__main__':
    unittest.main()