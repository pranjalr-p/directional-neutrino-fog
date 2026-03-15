import numpy as np
import scipy as scipy
import tqdm

'''
All code in the this section has been adapted from O'Hare. 
(see refs)
'''

def phiGen1(s, b, sigmaTheta, exposure):
    n_nu = len(b)
    temp = s + np.sum(b, axis=0)
    derList = [[exposure*np.sum(b[i]*x/temp) for x in b[i+1:]] for i in range(n_nu-1)]
    G1 = exposure*np.sum(s*s/temp)
    G2 = exposure*np.array([np.sum(s*x/temp) for x in b])
    G3 = np.zeros((n_nu,n_nu))
    for i in range(0, n_nu-1):
        G3[i,i+1:] = derList[i]
        G3[i+1:, i] = derList[i]
    diagTerm = 1/(sigmaTheta*sigmaTheta)+exposure*np.array([np.sum(x*x/temp) for x in b])
    G3 = G3+np.diag(diagTerm)
    res = G1 - G2@np.linalg.inv(G3)@G2
    return res

def MakeDL(m_vals,R_sig,R_nu,NuUnc,sigma_ref,sigma_min,sigma_max,ns,Ex_min,Ex_max,n_ex):
    nm = len(m_vals)
    Ex_vals = np.geomspace(Ex_min,Ex_max,n_ex)
    sig_vals = np.flipud(np.geomspace(sigma_min,sigma_max,ns))
    DL = np.zeros((nm,ns))
    for i in tqdm.tqdm(range(0,nm)):
        k0 = 0
        for j in range(0,ns):
            sig = sig_vals[j]
            P_prev = 0.0
            for k in range(k0,n_ex):
                if sum(Ex_vals[k]*R_sig[i,:]*sig/sigma_ref)>0:
                    if sum(Ex_vals[k]*R_sig[i,:]*sig/sigma_ref)>1e20:
                        DL[i,j] = np.nan
                    else:
                        nc = phiGen1(R_sig[i,:]*sig/sigma_ref,R_nu,NuUnc,Ex_vals[k])
                        P = scipy.stats.ncx2.sf(9,1,nc)
                        if P>0.9:
                            DL[i,j] = np.exp(np.interp(0.9,[P_prev,P],[np.log(Ex_vals[k-1]),np.log(Ex_vals[k])]))
                            k0 = np.maximum(0,k-1)
                            if k==n_ex:
                                k0 = 0
                            break
                        else:
                            P_prev = P
    return m_vals,sig_vals,DL


from scipy.ndimage import gaussian_filter1d
def Floor_2D(m,sig,DL,filt=False,filt_width=3,Ex_crit=1e20):
    n = np.size(m)
    ns = np.size(sig)
    Ex = DL.T*1
    #Ex[Ex>Ex_crit] = nan

    DY = np.zeros(shape=np.shape(Ex))
    for j in range(0,n):
        y = np.log10(Ex[:,j])
        if filt:
            y = gaussian_filter1d(gaussian_filter1d(y,sigma=3),filt_width)
            dy = np.gradient(y,np.log10(sig[2])-np.log10(sig[1]))
            dy = gaussian_filter1d(dy,filt_width)
        else:
            dy = np.gradient(y,np.log10(sig[2])-np.log10(sig[1]))
        
        DY[:,j] = dy

    NUFLOOR = np.zeros(shape=n)
    for j in range(0,n):
        for i in range(0,ns):
            if DY[ns-1-i,j]<=-2.0:
                i0 = ns-1-i
                i1 = i0+10
                NUFLOOR[j] = 10.0**np.interp(-2,DY[i0:i1+1,j],np.log10(sig[i0:i1+1]))
            
                mask = sig<NUFLOOR[j]
                dy = DY[mask,j]
                dy[dy>-2] = -2
                DY[mask,j] = dy       
                break
    DY = -DY
    return NUFLOOR,DY