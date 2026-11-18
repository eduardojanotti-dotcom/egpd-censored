"""
Extended generalized Pareto (EGP) models for right-censored data.

F(x) = H(G_theta(x)), where G_theta is a GPD(sigma, xi) with location 0 and H is a
perturbation distribution on [0, 1], estimated by one of:

  Beta_one_parameter           H(u) = u^k  (Beta(k, 1)), censored maximum likelihood
  Bernstein_interval_Censored  Bernstein polynomial with Kaplan-Meier weights
  Histogram_interval_Censored  histogram (Grenander-type sieve) with Kaplan-Meier weights

Semi_Ext_Gen_Pareto.fit runs the iterative fixed-point procedure described in the paper.
"""
import math
import warnings

import numpy as np
from scipy import stats
from scipy.stats import CensoredData, beta, genpareto


class Histogram_interval_Censored:

    def __init__(self):
        pass

    # ======================
    # PDF
    # ======================
    def pdf(self, x):

        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        mask = (x > 0) & (x <= 1)

        idx = np.searchsorted(self.edges, x[mask], side="right") - 1
        idx = np.clip(idx, 0, len(self.ω)-1)

        result[mask] = self.ω[idx] / self.widths[idx]

        return result

    # ======================
    # CDF
    # ======================
    def cdf(self, x):

        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        mask = (x > 0) & (x < 1)

        idx = np.searchsorted(self.edges, x[mask], side="right") - 1
        idx = np.clip(idx, 0, len(self.ω)-1)

        cumulative_mass = np.insert(np.cumsum(self.ω), 0, 0)

        left_edges = self.edges[idx]
        partial = (x[mask] - left_edges) / self.widths[idx]

        result[mask] = cumulative_mass[idx] + partial * self.ω[idx]

        result[x >= 1] = 1.0

        return result

    # ======================
    # LOG PDF
    # ======================
    def logpdf(self, x):
        return np.log(self.pdf(x))

    # ======================
    # SAMPLING
    # ======================
    def ppf(self, y):

        y = np.asarray(y)

        cumulative = np.cumsum(self.ω)
        idx = np.searchsorted(cumulative, y)

        prev_cum = np.insert(cumulative, 0, 0)[idx]

        u = (y - prev_cum) / self.ω[idx]

        return self.edges[idx] + u * self.widths[idx]

    def rvs(self, size):
        return self.ppf(np.random.uniform(size=size))

    # ======================
    # WEIGHT CONSTRUCTION
    # ======================
    def weights_approximation(self, kmf):

        edges = np.linspace(0, 1, self.m + 1)

        ω = []

        for i in range(len(edges)-1):
            up = kmf.cdf.evaluate(edges[i+1])
            dw = kmf.cdf.evaluate(edges[i])
            ω.append(up - dw)

        ω = np.array(ω)

        ω = np.asarray(ω, dtype=float)
        ω = np.clip(ω, 0, None)
        
        ω_sum = ω.sum()
        if ω_sum <= 0:
            raise ValueError("All weights are zero.")
        
        # ω = ω / ω_sum

        self.edges = edges
        self.ω = ω

        # enforce restriction
        self.enforce_last_bin_nonzero()
        self.put_mass_on_last_bin()

        self.widths = np.diff(self.edges)
        # self.ω = self.ω/(self.ω@self.widths)
        
        
        

    # ======================
    # FIT
    # ======================
    def fit(self, data, m=50):

        self.m = m
        km = stats.ecdf(data)

        self.kmf = km
        self.weights_approximation(km)

    # ======================
    # OPTION B MERGING
    # ======================
    def enforce_last_bin_nonzero(self, tol=1e-14):
        """
        Merge empty upper bins by enlarging the last
        nonempty interval.
        """

        ω = self.ω
        edges = self.edges

        positive = np.where(ω > tol)[0]
        if len(positive) == 0:
            raise ValueError("All bins empty.")

        r = positive[-1]

        if r == len(ω) - 1:
            return

        # merge all tail mass
        merged_mass = ω[r:].sum()

        ω_new = ω[:r+1].copy()
        ω_new[r] = merged_mass

        edges_new = edges[:r+2].copy()
        edges_new[-1] = 1.0  # extend last bin

        self.ω = ω_new
        self.edges = edges_new


    def put_mass_on_last_bin(self):
        """
        Put mass on last interval
        """

        ω = self.ω
        ω_new = ω[:].copy()
        ω_new[-1] = (1-ω[:-1].sum())
        self.ω = ω_new
        
        
        
class Beta_one_parameter:
    def __init__(self,):
        pass
        
    def cdf(self,x,a=None,b=None):
        if a is None:
            a = self.a 
        if b is None:
            b = self.b
        return beta.cdf(x,a,b)
        # return self.kmf.cumulative_density_at_times(x)
        
        
    def ppf(self,y):
        print("No implementation")
    
    def rvs(self,size):
        return self.ppf(np.random.uniform(size=size))
    
    
    def pdf(self,x,a=None,b=None):
        if a is None:
            a = self.a 
        if b is None:
            b = self.b
        return beta.pdf(x,a,b)
    
    
    
    def logpdf(self,x):    
        return np.log(self.pdf(x))


    def loglikelihood(self,x):
        print("No implementation")



        
    
    def fit(self,data):
        # H(u)=u^k is Beta(k,1) on [0,1]: fix b=1, loc=0 and scale=1
        fited = beta.fit(data,fb=1,floc=0,fscale=1)
        self.a = fited[0]
        self.b = fited[1]
        
        
        
        
        
        
class Bernstein_interval_Censored:
    def __init__(self,):
        pass
        
    def cdf(self,x):
        return self.G_hat_mn(x)
        
        
    def ppf(self,y):
        print("No implementation")
    
    def rvs(self,size):
        return self.ppf(np.random.uniform(size=size))
    
    
    def pdf(self,x):    
        return self.g_hat_mn(x)
    
    
    
    def logpdf(self,x):    
        return np.log(self.pdf(x))


    def loglikelihood(self,x):
        print("No implementation")



    def g_hat_mn(self,x):
        soma = 0
        for k in range(1,self.m+1):
            soma += self.ω[k-1]*beta.pdf(x,k,self.m-k+1)
        return soma
    
    
    def G_hat_mn(self,x):
        soma = 0
        for k in range(1,self.m+1):
            soma += self.ω[k-1]*beta.cdf(x,k,self.m-k+1)
        return soma
        
    def weights_approximation(self,data):
        ω_m = [] 
        G_n = stats.ecdf(data)

        for k in range(1,self.m+1):
            ω_m.append( G_n.cdf.evaluate(k/self.m) - G_n.cdf.evaluate((k-1)/self.m) )
        self.ω = ω_m
        if self.ω[-1] == 0:
            self.ω[-1] = 1 - self.G_hat_mn((self.m-1)/self.m)
        self.ω = self.ω/np.sum(self.ω)
        
        
    
    def fit(self,data,m=50):
        self.m = m
        self.weights_approximation(data,)
        
        
        
        
        
# ==========================================================
# Semi-parametric Extended GPD
# ==========================================================

class Semi_Ext_Gen_Pareto:
    def __init__(self):
        self.model = None
        self.Ѳ = []
        self.ξ = None
        self.σ = None
    
    def pdf(self,t,Ѳ=None):
        Ѳ = self.Ѳ[-1]
        return self.model.pdf(
            genpareto.cdf(t,c=Ѳ[0],loc=Ѳ[1],scale=Ѳ[2])
        )*genpareto.pdf(t,c=Ѳ[0],loc=Ѳ[1],scale=Ѳ[2])
    
    
    def cdf(self,t,Ѳ=None):
        Ѳ = self.Ѳ[-1]
        return self.model.cdf(
            genpareto.cdf(t,Ѳ[0],loc=Ѳ[1],scale=Ѳ[2])
        )

    def fit(self,data,m=None,debug=False,partial_debug=False,kernel="Bernstein"):

        # HARD RESET
        self.model = None
        self.Ѳ = []
        self.ξ = None
        self.σ = None
        # --------------------------------------------------
        # Kernel choice
        # --------------------------------------------------
        # if kernel == "Bernstein":
        #     KERNEL = Bernstein_interval

        if kernel == "Bernstein":
            KERNEL = Bernstein_interval_Censored
            m =m

        elif kernel == "Histogram":
            KERNEL = Histogram_interval_Censored
            m =m
            
        elif kernel == "Beta":
            KERNEL = Beta_one_parameter
            
            
        # --------------------------------------------------
        # Initial parametric fit
        # --------------------------------------------------
        

        est = genpareto.fit(data,floc=0)

        
        
        ξ = est[0]
        μ = est[1]
        σ = est[2]


        Ѳ = [(ξ,μ,σ)]

        cond = True
        eps = 1e-3


        

        
        times = np.concatenate([
            data._uncensored,
            data._right
        ])

        event = np.concatenate([
            np.ones_like(data._uncensored, dtype=int),   # event occurred
            np.zeros_like(data._right, dtype=int)        # censored
        ])

        # ==================================================
        # Fixed-point iteration
        # ==================================================
        n_iter = 0

        while cond:
            n_iter += 1
            Zi = genpareto.cdf(
                times,
                c=Ѳ[-1][0],
                scale=Ѳ[-1][2]
            )

            self.model = KERNEL()

            # SAFE FIT CALL
            Zi_event = Zi[event == 1]
            Zi_cens  = Zi[event == 0]
            

            data_temp_1 = CensoredData(uncensored=Zi_event,
                                right=Zi_cens)
            
            if m is None:
                self.model.fit(data_temp_1)
            else:
                self.model.fit(data_temp_1,m=m)
                
                
            # ----------------------------------------------
            # NUMERICALLY SAFE PIT
            # ----------------------------------------------
            GZi = self.model.cdf(Zi)
            # if kernel in ["Histogram"]:
            #     GZi = np.clip(GZi,0,1-1e-4)

            # if debug or partial_debug:
            #     aa = np.linspace(0,1,100)
            #     bb = self.model.pdf(aa)
            #     plt.plot(aa,bb)
            #     plt.grid()
            #     plt.show()
            GZi_base = GZi
            
            try:
                V = genpareto.ppf(GZi_base, c=Ѳ[-1][0], scale=Ѳ[-1][2])
                # ----------------------------------------------
                # PARAMETER UPDATE
                # ----------------------------------------------
                V_event = V[event == 1]
                V_cens  = V[event == 0]

                data_temp_2 = CensoredData(uncensored=V_event,
                                    right=V_cens)

            except Exception:
                failed = True
                vals = ([1 - 10**(-i) for i in range(10, 0, -1)]+ [i/100 for i in range(89, 0, -1)])
                for i in vals:
                    try:
                        upper = i  # vals already holds the clip levels 1-1e-10, ..., 0.01
                        GZi_adj = np.clip(GZi_base, 0, upper)
            
                        V = genpareto.ppf(GZi_adj, c=Ѳ[-1][0], scale=Ѳ[-1][2])
                        # ----------------------------------------------
                        # PARAMETER UPDATE
                        # ----------------------------------------------
                        V_event = V[event == 1]
                        V_cens  = V[event == 0]

                        data_temp_2 = CensoredData(uncensored=V_event,
                                            right=V_cens)
                        warnings.warn(f"GPD quantile not finite; pseudo-observations truncated at {i}")
                        failed = False
                        break
                    except Exception:
                        pass
            
                if failed:
                    raise RuntimeError("GPD PPF failed even after progressive clipping.")
            
                


            est = genpareto.fit(data_temp_2,floc=0)
            ξ_new = est[0]
            σ_new = est[2]

            # ----------------------------------------------
            # Convergence
            # ----------------------------------------------
            if math.isnan(abs(ξ_new-Ѳ[-1][0])):
                raise FloatingPointError("shape update is NaN")

            if abs(ξ_new-Ѳ[-1][0]) < eps:
                cond = False
            if n_iter > 100:  # at most 101 updates of (xi, sigma)
                # print(f"ERROR, number of iteration achieved maximum {n_iter}")
                cond = False

            Ѳ.append((ξ_new,0,σ_new,))

            self.ξ = ξ_new
            self.σ = σ_new

            # if debug or partial_debug:
            #     print("σ:",σ_new)
            #     print("ξ:",ξ_new)
            
        self.Ѳ=Ѳ
        return Ѳ[-1]
