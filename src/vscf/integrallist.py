import numpy as np

def delta(i, j):
    return np.where(i == j, 1, 0)

def get_psi_q0_psi(Nlevels):
    return np.eye(Nlevels, dtype=np.float64)

def get_psi_q1_psi(Nlevels):
    psi_q1_psi = np.zeros((Nlevels, Nlevels))
    for m in range(Nlevels):
        for n in range(Nlevels):
            psi_q1_psi[m,n] =  delta(m,n+1) * np.sqrt(m/2) \
                             + delta(m,n-1) * np.sqrt((m+1)/2)
    return np.array(psi_q1_psi, dtype=np.float64)

def get_psi_q2_psi(Nlevels):
    psi_q2_psi = np.zeros((Nlevels, Nlevels))
    for m in range(Nlevels):
        for n in range(Nlevels):
            psi_q2_psi[m,n] =  delta(m,n+2) * (1/2) * np.sqrt((m-1) * m) \
                             + delta(m,n)   * (m + 1/2) \
                             + delta(m,n-2) * (1/2) * np.sqrt((m+1) * (m+2))
    return np.array(psi_q2_psi, dtype=np.float64)

def get_psi_q3_psi(Nlevels):
    psi_q3_psi = np.zeros((Nlevels, Nlevels))
    for m in range(Nlevels):
        for n in range(Nlevels):
            psi_q3_psi[m,n] =  delta(m,n+3) * (1/(2*np.sqrt(2))) * np.sqrt((m-2) * (m-1) * m) \
                             + delta(m,n+1) * (1/(2*np.sqrt(2))) * 3 * np.sqrt(m**3) \
                             + delta(m,n-1) * (1/(2*np.sqrt(2))) * 3 * np.sqrt((m+1)**3) \
                             + delta(m,n-3) * (1/(2*np.sqrt(2))) * np.sqrt((m+1) * (m+2) * (m+3))
    return np.array(psi_q3_psi, dtype=np.float64)

def get_psi_q4_psi(Nlevels):
    psi_q4_psi = np.zeros((Nlevels, Nlevels))
    for m in range(Nlevels):
        for n in range(Nlevels):
            psi_q4_psi[m,n] =  delta(m,n+4) * (1/4) * np.sqrt((m-3) * (m-2) * (m-1) * m) \
                             + delta(m,n+2) * (1/2) * (2*m-1) * np.sqrt((m-1) * m) \
                             + delta(m,n)   * (3/4) * (2*m**2 + 2*m + 1) \
                             + delta(m,n-2) * (1/2) * (2*m+3) * np.sqrt((m+1) * (m+2)) \
                             + delta(m,n-4) * (1/4) * np.sqrt((m+1) * (m+2) * (m+3) * (m+4))
    return np.array(psi_q4_psi, dtype=np.float64)

def get_psi_qT_psi(Nlevels):
    psi_qT_psi = np.zeros((Nlevels, Nlevels))
    for m in range(Nlevels):
        for n in range(Nlevels):
            psi_qT_psi[m,n] =  (-1/2) * delta(m,n+2) * (1/2) * np.sqrt((m-1) * m) \
                             + ( 1/2) * delta(m,n)   * (m + 1/2) \
                             + (-1/2) * delta(m,n-2) * (1/2) * np.sqrt((m+1) * (m+2))
    return np.array(psi_qT_psi, dtype=np.float64)

def get_psi_q_psi(Nlevels):
    psi_q0_psi = get_psi_q0_psi(Nlevels)
    psi_q1_psi = get_psi_q1_psi(Nlevels)
    psi_q2_psi = get_psi_q2_psi(Nlevels)
    psi_q3_psi = get_psi_q3_psi(Nlevels)
    psi_q4_psi = get_psi_q4_psi(Nlevels)
    psi_qT_psi = get_psi_qT_psi(Nlevels)
    return psi_q0_psi, psi_q1_psi, psi_q2_psi, psi_q3_psi, psi_q4_psi, psi_qT_psi

