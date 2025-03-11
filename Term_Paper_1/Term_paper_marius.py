# %% [markdown]
# ***
# # Part 1
# ***

# %%
# Import dataclass from the dataclasses module
from dataclasses import dataclass


# Define a data class to store Parameters
@dataclass
class Parameters:
    """
    Parameters for the overlapping generations model (OLG)
    """
    # Part 1 - 3 - OLG Model Parameters
    beta: float = 0.96**30              # Discount factor
    gamma: float = 5.0                  # Relative risk aversion
    alpha: float = 0.36                 # Capital share of output (production)
    delta: float = 1 - 0.94**30         # Capital depreciation
    z: float = 1.0                      # Total factor productivity (TFP)
    
    # Part 4 - PAYGO system
    tau: float = 0.0                    # Payroll tax rate (Default is  0.0 --> No PAYGO system)
    
    # Part 5 - AR (1) Parameters
    rho: float = 0.95**30               # Autocorrelation 
    sigma_2: float = 0.05**2            # Conditional variance 
    mu: float = -1/2*sigma_2/(1-rho)    # Conditional mean

    epsilon_grid: np.ndarray = None     # Discretized shock values
    epsilon_prob: np.ndarray = None     # Corresponding probabilities

# Create two instances of the Parameters class
par = Parameters() # tau = 0.0
par_paygo = Parameters(tau=0.1) # tau = 0.1

# %%
# Define a function to map k to factor prices r and w using the first-order conditions
def compute_prices(k, par: Parameters):
    """
    Return factor prices for a given capital-labor ratio and parameters.

    Parameters
    ----------
    k : float
        Capital-labor ratio.
    par : Parameters
        Instance of the Parameters class including model parameters for the OLG model.

    Returns
    -------
    r : float
        Return on capital after depreciation (interest rate)
    w : float
        Wage rate
    """
    # Return on capital after depreciation (interest rate)
    r = par.alpha * par.z * (k ** (par.alpha - 1)) - par.delta

    # Wage rate
    w = (1 - par.alpha) * par.z * (k**par.alpha)

    return r, w

# %%
# Define a function to compute the Euler equation error
def euler_err(k_next, k, par: Parameters):
    """
    Compute the Euler equation error for a given capital stock today and tomorrow.

    Parameters
    ----------
    k_next : float
        Capital in the next period.
    k : float
        Capital in the current period.
    par : Parameters
        An instance of the Parameters class.

    Returns
    -------
    float
        Euler equation error.
    """
    # Compute factor prices for the todays period
    _, w = compute_prices(k, par)

    # Compute factor prices for the next period
    r_next, w_next = compute_prices(k_next, par)

    # Asset market clearing condition
    a = k_next

    # Pension paymets
    p_next = par.tau * w_next

    # Consumption by the young
    c_y = (1 - par.tau) * w - a
    # Consumption by the old
    c_o = (1 + r_next) * a + p_next

    # Left-hand side of the Euler equation
    lhs = c_y ** (-par.gamma)

    # Right-hand side of the Euler equation
    rhs = par.beta * (1 + r_next) * (c_o ** (-par.gamma))

    # Euler equation error
    err = lhs - rhs

    return err

# %%
# Import nump for numerical operations
import numpy as np

# Create a grid of capital values, k, on the interval [0.05, 0.2]
k_gird = np.linspace(0.05, 0.2, 100)

# Create an empty array to store the Euler equation error
euler_err_grid = np.empty_like(k_gird)

# Compute the Euler equation error for each value of k assuming k_next = k
euler_err_grid = euler_err(k_gird, k_gird, par)

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Plot the Euler equation error as a function of the capital stock
plt.figure(figsize=(8, 5))
plt.plot(k_gird, euler_err_grid, c="red", lw=1, label="Euler equation error")
plt.axhline(0, c="black", ls="--", lw=0.5)
plt.xlabel("Capital stock (k)")
plt.ylabel("Euler equation error")
plt.title("Euler equation error as a function of the capital stock")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# %% [markdown]
# 
# ### Why is the graph upward sloping?
# 

# %% [markdown]
# ***
# # Part 2
# ***

# %%
# Defina a data class to store the steady state quantities and prices
@dataclass
class SteadyState:
    par: Parameters = None  # Parameters used to compute equilibrium
    c_y: float = None  # Consumption when young
    c_o: float = None  # Consumption when old
    a: float = None  # Savings when young
    s: float = None  # Savings rate when young
    r: float = None  # Interest rate (return on capital)
    w: float = None  # Wage rate
    K: float = None  # Aggregate capital stock
    L: float = None  # Aggregate labor demand
    I: float = None  # Aggregate investment
    Y: float = None  # Aggregate output

# %% [markdown]
# ### <ins> **Finding the Optimal Savings Rate s without and with a PAYGO System** </ins>
# 
# We start with the Euler equation, which ensures optimal consumption allocation over time:
# 
# $
# u'(c_{y,t}) = \beta (1 + r) u'(c_{o,t+1})
# $
# 
# 
# #### ***1. Substituting CRRA Utility Function***
# Using the CRRA utility function:
# 
# $
# u'(c) = c^{-\gamma}
# $
# 
# the Euler equation becomes:
# 
# $
# ((1 - \tau)w_t - a_t)^{-\gamma} = \beta (1 + r) \left( (1 + r)a_t + \tau w_t \right)^{-\gamma}
# $
# 
# 
# #### ***2. Expressing in Terms of the Savings Rate***
# We define the savings rate as:
# 
# $
# s = \frac{a_t}{(1 - \tau)w_t}
# $
# 
# Rewriting consumption expressions in terms of  $s$:
# 
# - *Young period consumption:* 
# 
#   $
#   c_{y,t} = (1 - \tau)w_t - a_t = (1 - \tau)(1 - s)w_t
#   $
# 
# - *Old period consumption:*  
# 
#   $
#   c_{o,t+1} = (1 + r)a_t + \tau w_t = (1 + r)s(1 - \tau)w_t + \tau w_t
#   $
#   
# <br> </br>
# Substituting these into the **Euler equation**:
# 
# $
# ((1 - \tau)(1 - s)w_t)^{-\gamma} = \beta (1 + r) \left( (1 + r)s(1 - \tau)w_t + \tau w_t \right)^{-\gamma}
# $
# <br> </br>
# Canceling $w_t$ on both sides:
# 
# $
# ((1 - \tau)(1 - s))^{-\gamma} = \beta (1 + r) \left( (1 + r)s(1 - \tau) + \tau \right)^{-\gamma}
# $
# 
# 
# #### ***3. Solving for $s$***
# Since both sides are raised to $-\gamma$, we take reciprocals:
# 
# $
# (1 - \tau)(1 - s) = \left[ \beta (1 + r) \right]^{-\frac{1}{\gamma}} \left[ (1 + r)s(1 - \tau) + \tau \right]
# $
# <br> </br>
# Rearrange to solve for $s$:
# 
# $
# s = \frac{(1 - \tau) - \beta^{-\frac{1}{\gamma}}(1 + r)^{-\frac{1}{\gamma}} \tau}
# {(1 - \tau) \left[ 1 + \beta^{-\frac{1}{\gamma}} (1 + r)^{1 - \frac{1}{\gamma}} \right]}
# $
# 
# This is the ***optimal savings rate*** accounting for a ***PAYGO tax system***.
# 
# - If **$\tau = 0$**, the formula *"becomes"*  the standard savings rate equation $ \Rightarrow    s = \frac{1}{1 + \beta^{-\frac{1}{\gamma}} (1+r)^{1-\frac{1}{\gamma}}}$
# 
# - If **$\tau  \ne 0$**, the optimal savings rate is given by the following equation $ \Rightarrow     s = \frac{(1 - \tau) - \beta^{-\frac{1}{\gamma}}(1 + r)^{-\frac{1}{\gamma}} \tau}{(1 - \tau) \left[ 1 + \beta^{-\frac{1}{\gamma}} (1 + r)^{1 - \frac{1}{\gamma}} \right]}$

# %%
def compute_savings_rate(r, par: Parameters):
    """
    Compute the savings rate using the analytical solution
    to the household problem, either with or without a PAYGO system.

    Parameters
    ----------
    r : float
        Return on capital after depreciation (interest rate)
    par : Parameters
        Parameters for the given problem

    Returns
    -------
    s : float
        Optimal Savings rate
    """
    # Define A = beta^(-1/gamma) * (1+r)^(-1/gamma) to simplify the expression
    A = par.beta ** (-1.0 / par.gamma) * (1.0 + r) ** (-1.0 / par.gamma)

    # Compute the savings rate using the analytical solution
    s = ((1 - par.tau) - A * par.tau) / ((1 - par.tau) * (1 + A * (1 + r)))

    return s

# %%
# Import the root_scalar function from scipy.optimize
from scipy.optimize import root_scalar


# Define a function to compute the steady state equilibrium using root finding
def compute_steady_state(par: Parameters):
    """
    Compute the steady-state equilibrium for the OLG model.

    Parameters
    ----------
    par : Parameters
        Parameters for the given problem

    Returns
    -------
    eq : SteadyState
        Steady state equilibrium of the OLG model
    """
    res = root_scalar(
        lambda k: euler_err(k, k, par),  # k_next = k
        bracket=(0.01, 0.2),  # search interval
    )

    if not res.converged:
        print("Equilibrium root-finder did not converge.")

    # Equilibrium capital stock
    K = res.root

    # Create instance of SteadyState with explicit L
    eq = SteadyState(par=par, K=K, L=1)

    # Compute the equilibrium factor prices
    eq.r, eq.w = compute_prices(eq.K / eq.L, par)

    # Investment in steady state
    eq.I = (eq.K * par.delta)  # Since we assume k_next = k --> I = k_next - (1-delta)k = k(1 - (1 - delta)) = k*delta

    # Equilibrium household decisions
    eq.s = compute_savings_rate(eq.r, par)

    # Pension payments
    p = par.tau * eq.w

    # Compute consumption when young and old
    eq.a = eq.s * eq.w * (1 - par.tau)
    eq.c_y = (1 - par.tau) * eq.w - eq.a
    eq.c_o = (1 + eq.r) * eq.a + p

    # Compute output (Include L even though it is 1)
    eq.Y = par.z * eq.K**par.alpha * eq.L ** (1 - par.alpha)

    # Aggregate consumption
    C = eq.c_y + eq.c_o

    # Check that goods market clearing holds using : Y = C + I
    assert (abs(eq.Y - (C + eq.I)) < 1.0e-8)  # Assert that the difference is less than 1.0e-8

    return eq

# %%
# Function to print the steady state equilibrium
def print_steady_state(eq: SteadyState):
    """
    Print equilibrium prices, allocations, and excess demand.

    Parameters
    ----------
    eq : SteadyState
        SteadyState of the OLG model
    """
    print("Steady-state equilibrium:")
    print(f"------------------------------------------------")
    print("  Households:")
    print(f"    Consumption when young (c_y) =  {eq.c_y:.5f}")
    print(f"    Consumption when old (c_o) =    {eq.c_o:.5f}")
    print(f"    Savings (a) =                   {eq.a:.5f}")
    print(f"    Savings rate (s) =              {eq.s:.5f}")
    print(f"------------------------------------------------")
    print("  Firms:")
    print(f"    Capital (K) =                   {eq.K:.5f}")
    print(f"    Labor (L) =                     {eq.L:.5f}")
    print(f"    Output (Y) =                    {eq.Y:.5f}")
    print(f"------------------------------------------------")
    print("  Prices:")
    print(f"    Interest rate (r) =             {eq.r:.5f}")
    print(f"    Wage rate (w) =                 {eq.w:.5f}")
    print(f"------------------------------------------------")
    print("  Market Clearing Conditions:")
    print(f"    Capital market:                 {eq.K - eq.a:.2e}")
    print(f"    Goods market:                   {eq.Y - (eq.c_y + eq.c_o) - eq.I :.2e}")
    print(f"------------------------------------------------")

# %%
# Compute the steady state equilibrium
eq = compute_steady_state(par)

# Print the steady state equilibrium
print_steady_state(eq)

# %% [markdown]
# ***
# # Part 3
# ***

# %%
# Define a data class to store the simulated time series
@dataclass
class Simulation:
    """
    Stores time series data for transition dynamics.
    """

    c_y: np.ndarray = None  # Time series for consumption when young
    c_o: np.ndarray = None  # Time series for consumption when old
    a: np.ndarray = None  # Time series for savings when young
    s: np.ndarray = None  # Time series for savings rate when young
    r: np.ndarray = None  # Time series for interest rate (return on capital)
    w: np.ndarray = None  # Time series for wages
    K: np.ndarray = None  # Time series for aggregate capital stock
    Y: np.ndarray = None  # Time series for aggregate output
    z: np.ndarray = None  # Time series for TFP
    tau: np.ndarray = None  # Time series for payroll tax rate

# %%
# Define a function to initialize the simulation and allocate arrays for time series
def initialize_sim(T, eq: SteadyState = None):
    """
    Initialize simulation instance (allocate arrays for time series).

    Parameters
    ----------
    T : int
        Number of periods to simulate
    eq : SteadyState, optional
        Steady-state equilibrium to use for initial period
    """
    # Initialize simulation instance
    sim = Simulation()

    # Initialize time series arrays
    sim.c_y = np.zeros(T + 1)
    sim.c_o = np.zeros(T + 1)
    sim.a = np.zeros(T + 1)
    sim.s = np.zeros(T + 1)
    sim.r = np.zeros(T + 1)
    sim.w = np.zeros(T + 1)
    sim.K = np.zeros(T + 1)
    sim.Y = np.zeros(T + 1)
    sim.z = np.zeros(T + 1)
    sim.tau = np.zeros(T + 1)

    # If eq is provided, set initial values to steady state
    if eq is not None:
        sim.c_y[0] = eq.c_y
        sim.c_o[0] = eq.c_o
        sim.a[0] = eq.a
        sim.s[0] = eq.s
        sim.r[0] = eq.r
        sim.w[0] = eq.w
        sim.K[0] = eq.K
        sim.Y[0] = eq.Y
        sim.z[0] = eq.par.z
    
    # Start the economy withou a PAYGO system
    sim.tau[0] = 0.0

    return sim

# %%
import copy
    
# Define a function to simulate the transition dynamics of the OLG model
def simulate_olg(K1, T, eq: SteadyState, par: Parameters):
    """
    Simulate the transition dynamics of the OLG model starting from steady state
    when a shock to capital realizes in period 1.

    Parameters
    ----------
    K1 : float
        Initial capital stock in period 1.
    T : int
        Number of periods to simulate.
    eq : SteadyState
        Initial steady-state equilibrium of the OLG model before the shock.
    par : Parameters
        Model parameters including PAYGO system.

    Returns
    -------
    sim : Simulation
        Simulation object containing the time series for each variable.
    """

    par_sim = copy.copy(par)  # Copy parameters to avoid modifying the original
    sim = initialize_sim(T, eq)  # Initialize simulation structure
    sim.K[1] = K1  # Set the capital stock in period 1 (Optional shock)
    sim.z[:] = par_sim.z  # Set the TFP to initial value for all periods

    # Iterate over time periods
    for t in range(1, T + 1):

        # Dynamically choose root-finding interval based on previous period's capital
        low_k = 0.5 * sim.K[t]  # Minimum 50% of previous capital or 0.01
        high_k = 2.0 * sim.K[t]  # Maximum 200% of previous capital or 0.2

        if t < T:
            # Solve for k_next by finding the root of the Euler error
            res = root_scalar(lambda k_next: euler_err(k_next, sim.K[t], par_sim),
                bracket=(low_k, high_k),)

            if not res.converged:
                print(f"WARNING: Root-finding failed at t={t}, using fallback value")
                
            # Update capital stock
            sim.K[t + 1] = res.root

        # Update interest rate and wage rate
        sim.r[t], sim.w[t] = compute_prices(sim.K[t], par_sim)

        # Calculate savings rate
        sim.s[t] = compute_savings_rate(sim.r[t], par_sim)

        # Compute pension payments
        p = par_sim.tau * sim.w[t]

        # Compute consumption when young and old
        sim.a[t] = sim.s[t] * sim.w[t] * (1 - par_sim.tau)
        sim.c_y[t] = (1 - par_sim.tau) * sim.w[t] - sim.a[t]
        sim.c_o[t] = (1 + sim.r[t]) * sim.a[t - 1] + p  # Note: Here we calculate c_0(t) not c_0(t+1), meaning we use a[t-1] and not r(t+1) as we did in the Euler error function

        # Update tau and output
        sim.tau[t] = par_sim.tau # Purpose: To keep track of the PAYGO system
        sim.Y[t] = sim.z[t] * sim.K[t] ** par_sim.alpha  # Production (L=1)

        # Check goods market clearing
        demand = sim.c_y[t] + sim.c_o[t] + sim.a[t]
        supply = sim.Y[t] + (1 - par_sim.delta) * sim.K[t]
        market_clearing_diff = abs(demand - supply)

        if market_clearing_diff > 1.0e-8:
            print(f"WARNING: Goods market clearing failed at t={t}, with diff={market_clearing_diff:.6e}")

    return sim

# %%
# Shock: Assuming the capital falls to half of steady state in period 1
K1 = eq.K / 2

# Simulate the transition dynamics for T = 20 periods --> [0, 1, 2, ..., 20] meaning 21 values
sim = simulate_olg(K1, T = 20, eq = eq, par = par)

# Goods market clearing condition will fail in the first period due to the shock
# once the shock is absorbed, the goods market will clear
# (for our case: not enough periods to reach steady state)

# %%
from matplotlib.ticker import PercentFormatter

def plot_simulation(eq, sim, eq_new=None, deviations=True, filename=None):
    """
    Plot the simulated time series for Y, K, w, r, s in a 3x2 grid.
    The first three (Y, K, w) are shown as percent deviations from their
    initial steady-state values if deviations=True. The last two (r, s)
    are shown in levels.

    Parameters
    ----------
    eq : SteadyState
        Initial steady-state equilibrium. Used for reference lines
        and to compute deviations if 'deviations=True'.
    sim : Simulation
        The simulation object containing time series data.
    eq_new : SteadyState, optional
        If provided, also plot a dashed horizontal line for the
        new steady-state level in each subplot.
    deviations : bool, default True
        If True, plot Y, K, w as % deviations from the old steady state;
        if False, plot them in levels.
    filename : str, optional
        If provided, the figure is saved to this file.
    """
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True)

    # Turn off the last subplot if z is not in the simulation
    if sim.z is None:
        axes[2, 1].axis("off")

    # Keyword arguments for the main time-series plots
    kwargs_line = {
        "marker": "o" if len(sim.Y) < 30 else None,
        "markersize": 4,
        "color": "red",
        "label": "Simulation",
    }

    # Keyword arguments for the old (initial) steady-state lines
    kwargs_init = {
        "color": "black",
        "linewidth": 0.8,
        "linestyle": "--",
        "label": "Steady State" if eq_new is not None else "Initial SS",
    }

    # Keyword arguments for the new steady-state lines
    kwargs_new = {
        "color": "blue",
        "linewidth": 0.8,
        "linestyle": "--",
        "label": "New SS",
    }
    # ------------------------------------------------
    # 0) Preparation: Deviations and smoothing functions
    # ------------------------------------------------

    # Define a function to compute % deviations
    def pct_dev(series, ss_value):
        """
        Compute percent deviations of series from ss_value.

        Parameters
        ----------
        series : array-like
            The series to compute deviations for.
        ss_value : float
            The steady-state value to compute deviations from.

        Returns
        -------
        array-like
            Percent deviations of series from ss_value.
        """
        if deviations:
            return 100 * (series / ss_value - 1)
        else:
            return series

    # Smooth series to correct for numerical floating point errors
    def smooth_series(series, threshold=1e-6):
        """
        Snap series[i] to series[i-1] if their difference
        is below threshold.

        Parameters
        ----------
        series : array-like
            The series to smooth.
        threshold : float, optional
            The threshold below which to snap values together.
        Returns
        -------
        array-like
            The smoothed series.
        """
        for i in range(1, len(series)):
            if abs(series[i] - series[i - 1]) < threshold:
                series[i] = series[i - 1]
        return series

    # ------------------------------------------------
    # 1) Smooth all simulation series
    # ------------------------------------------------
    sim.Y = smooth_series(sim.Y, threshold=1e-6)
    sim.K = smooth_series(sim.K, threshold=1e-6)
    sim.w = smooth_series(sim.w, threshold=1e-6)
    sim.r = smooth_series(sim.r, threshold=1e-6)
    sim.s = smooth_series(sim.s, threshold=1e-6)

    # ------------------------------------------------
    # 2) Y_t: % dev if deviations = True, else level
    # ------------------------------------------------
    ax = axes[0, 0]
    yvals = pct_dev(sim.Y, eq.Y)
    ax.plot(yvals, **kwargs_line)
    # Steady-state reference line
    ref_line = 0 if deviations else eq.Y
    ax.axhline(ref_line, **kwargs_init)
    # If new SS is provided
    if eq_new is not None:
        ref_line_new = 0 if deviations else eq_new.Y
        if deviations:
            ref_line_new = 100 * (eq_new.Y / eq.Y - 1)
        ax.axhline(ref_line_new, **kwargs_new)
    ax.set_title(r"$Y_t$")
    ax.set_ylabel("Deviations from initial SS" if deviations else "Level")

    # ------------------------------------------------
    # 3) K_t: % dev if deviations=True, else level
    # ------------------------------------------------
    ax = axes[0, 1]
    yvals = pct_dev(sim.K, eq.K)
    ax.plot(yvals, **kwargs_line)
    # Steady-state reference line
    ref_line = 0 if deviations else eq.K
    ax.axhline(ref_line, **kwargs_init)
    # If new SS is provided
    if eq_new is not None:
        ref_line_new = 0 if deviations else eq_new.K
        if deviations:
            ref_line_new = 100 * (eq_new.K / eq.K - 1)
        ax.axhline(ref_line_new, **kwargs_new)
    ax.set_title(r"$K_t$")

    # ------------------------------------------------
    # 4) w_t: % dev if deviations=True, else level
    # ------------------------------------------------
    ax = axes[1, 0]
    yvals = pct_dev(sim.w, eq.w)
    ax.plot(yvals, **kwargs_line)
    # Steady-state reference line
    ref_line = 0 if deviations else eq.w
    ax.axhline(ref_line, **kwargs_init)
    # If new SS is provided
    if eq_new is not None:
        ref_line_new = 0 if deviations else eq_new.w
        if deviations:
            ref_line_new = 100 * (eq_new.w / eq.w - 1)
        ax.axhline(ref_line_new, **kwargs_new)
    ax.set_title(r"$w_t$")
    ax.set_ylabel("Deviations from initial SS" if deviations else "Level")

    # ------------------------------------------------
    # 5) r_t (always in levels)
    # ------------------------------------------------
    ax = axes[1, 1]
    ax.plot(sim.r, **kwargs_line)
    ax.axhline(eq.r, **kwargs_init)
    if eq_new is not None:
        ax.axhline(eq_new.r, **kwargs_new)
    ax.set_title(r"$r_t$ (levels)")
    ax.set_xlabel("Period")
    ax.tick_params(axis="x", labelbottom=True)

    # ------------------------------------------------
    # 6) s_t (always in levels)
    # ------------------------------------------------
    ax = axes[2, 0]
    ax.plot(sim.s, **kwargs_line)
    ax.axhline(eq.s, **kwargs_init)
    if eq_new is not None:
        ax.axhline(eq_new.s, **kwargs_new)
    ax.set_title(r"$s_t = \frac{a_t}{w_t}$ (levels)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Level")
    
    # ------------------------------------------------
    # 7) z_t (always in levels)
    # ------------------------------------------------
    ax = axes[2, 1]
    ax.plot(sim.z, **kwargs_line)
    ax.axhline(eq.par.z, **kwargs_init)
    if eq_new is not None:
        ax.axhline(eq_new.par.z, **kwargs_new)
    ax.set_title(r"$z_t$ (levels)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Level")

    # ------------------------------------------------
    # 8) Apply settings common to all axes
    # ------------------------------------------------

    # Apply settings common to all axes
    if deviations:
        # Format the first three subplots as percent
        for row, col in [(0, 0), (0, 1), (1, 0)]:
            axes[row, col].yaxis.set_major_formatter(PercentFormatter(decimals=0))

    axes[0, 0].legend(loc="right")
    fig.tight_layout()

    # Optional save
    if filename:
        plt.savefig(filename)

# %%
# Plot the transition dynamics of the OLG model
plot_simulation(eq, sim)  # No new steady state

# %% [markdown]
# ***
# # Part 4
# ***

# %%
# For paygo system we will use par_paygo, where tau = 0.1

# Compute the new steady state equilibrium
eq_pension = compute_steady_state(par_paygo)

# Print the new steady state equilibrium

print_steady_state(eq_pension)

# %% [markdown]
# In a pay‐as‐you‐go (PAYGO) system, today’s young workers are taxed and that revenue goes to pay pensions for the currently old. Because young households anticipate that they will receive a pension when they’re old, they have less incentive to save on their own.
# 
# Lower saving rate. The tax 
# 𝜏
# τ reduces the amount of after‐tax wage available for saving, and households also know they will get the pension benefit later. Both effects reduce their optimal private saving.
# Lower capital stock. In a two‐period model with one unit of young each period, less saving by the young directly translates to a smaller stock of capital in the next period’s production.
# The intuitive reason is straightforward: if the government guarantees some pension benefit, each household does not need to put aside as much of its own income, so the total (private) capital accumulation falls.
# 
# When the PAYGO system reduces households’ private saving and thereby the aggregate capital stock, the marginal product of capital rises (since capital is now scarcer). Because the interest rate r is closely tied to the marginal productivity of capital, the lower capital stock leads to a higher steady‐state interest rate.
# 
# In short:
# 
# Lower capital stock ⟹ Higher marginal product of capital ⟹ Higher r.

# %%
# Simulate the transition dynamics for T = 20 periods
sim_olg_pension = simulate_olg(eq.K, T=20, eq = eq, par = par_paygo)  # No shock in capital

# %%
# Verify that tau is updated in the simulation
print(sim_olg_pension.tau)

# %%
# Plot the transition dynamics of the OLG model with PAYGO system
plot_simulation(eq, sim_olg_pension, eq_pension)

# %% [markdown]
# ***
# # Part 5
# ***

# %%
# Import the hermegauss function to get Gauss-Hermite nodes and weights
from numpy.polynomial.hermite_e import hermegauss

epsilon_grid, epsilon_prob = hermegauss(5)  # Get Gauss-Hermite nodes and weights
epsilon_prob /= np.sqrt(2 * np.pi)          # Normalize probabilities

# Assign for both instances of the Parameters class (par and par_paygo)
par.epsilon_grid = epsilon_grid             # Assign nodes to the Parameters instance
par.epsilon_prob = epsilon_prob             # Assign weights to the Parameters instance
par_paygo.epsilon_grid = epsilon_grid        # Assign nodes to the Parameters instance
par_paygo.epsilon_prob = epsilon_prob        # Assign weights to the Parameters instance

# %%
def simulate_ar1(z0, T, seed, par: Parameters):
    """
    Simulate an AR(1) process for TFP.

    Parameters
    ----------
    z0 : float
        Initial TFP value.
    T : int
        Number of periods to simulate.
    seed : int
        Seed for the random number generator.
    par : Parameters
        An instance of the Parameters class containing additional AR(1) parameters:
          - par.mu        (float): Intercept in the log-AR(1)
          - par.rho       (float): Persistence parameter in the log-AR(1)
          - par.sigma_2   (float): Variance of the log-AR(1) innovation
          - par.epsilon_grid (np.ndarray): Discrete support for shocks
          - par.epsilon_prob (np.ndarray): Probabilities for each shock in epsilon_grid

    Returns
    -------
    np.ndarray
        Array (length T+1) of simulated TFP values.
    """
    # Create an array to store the simulated TFP values
    z = np.zeros(T + 1) 
    
    # Set the initial value of TFP
    z[0] = z0 
    
    # Create a RNG instance
    rng = np.random.default_rng(seed=seed)
    
    # Simulate the AR(1) process over time
    for t in range(T):
        # Draw one shock from the discrete approximation
        epsilon = rng.choice(par.epsilon_grid, p=par.epsilon_prob)
        
        # Update log of TFP
        log_z = (par.mu + par.rho * np.log(z[t]) + np.sqrt(par.sigma_2) * epsilon)

        # Convert log TFP back to levels
        z[t + 1] = np.exp(log_z)

    return z


# %%
# Parameters for the test
z0 = par_paygo.z  # Steady state TFP value 
T = 100_000  # Number of periods
seed = 1234  # Random seed for reproducibility

# Simulate TFP time series
tfp_sim = simulate_ar1(z0, T, seed, par)

# Compute the mean of simulated TFP values
tfp_mean = np.mean(tfp_sim)

# Plot the simulated TFP series
plt.figure(figsize=(10, 5))
plt.plot(tfp_sim, label="Simulated TFP", alpha=0.75, c="red", lw=1)
plt.axhline(1, color='black', linestyle='--', lw=0.5, label="Steady-state TFP (z=1)")
plt.xlabel("Time")
plt.ylabel("TFP Level")
plt.title("Simulated AR(1) Process for TFP")
plt.grid(True, linestyle="--", alpha=0.2)
plt.legend(loc = 'upper right')

# Expected Mean TFP value
print(f"Mean TFP (Expected ≈ 1): {tfp_mean:.4f}")

# %% [markdown]
# ***
# # Part 6
# ***

# %%
# Modify the euler_err function so that TFP follows an AR(1) process
def euler_err_ar1(k_next, k, z, par: Parameters):
    """
    Compute the Euler equation error if log TFP follows an AR(1) process.

    Parameters
    ----------
    k_next : float
        Capital in the next period.
    k : float
        Capital in the current period.
    z : float
        TFP in the current period.
    par : Parameters
        An instance of the Parameters class.
        
    Returns
    -------
    float
        Euler equation error.
    """
    # Left-hand side of the Euler equation
    _, w = compute_prices(k, par)           # Compute factor prices for the todays period
    a = k_next                              # Asset market clearing condition
    c_y = (1 - par.tau) * w - a             # Consumption by the young
    lhs = c_y ** (-par.gamma)               # Left hand side
    
    # Right-hand side of the Euler equation
    expected_rhs = 0                        # Set initial value for expected RHS
    
    for eps, prob in zip(par.epsilon_grid, par.epsilon_prob):
        # Next-period TFP from AR(1)
        log_z_next = par.mu + par.rho * np.log(z) + np.sqrt(par.sigma_2)*eps
        z_next = np.exp(log_z_next)

        # Factor prices next period
        par_ar1 = copy.copy(par)           # Copy parameters to avoid modifying the original
        par_ar1.z = z_next                 # Update TFP so that it can be used in compute_prices()
        r_next, w_next = compute_prices(k_next, par_ar1)  # Compute factor prices for the next period

        p_next = par.tau * w_next           # pension
        c_o = (1 + r_next) * a + p_next     # Consumption by the old

        # Next-period marginal utility term
        rhs_component = (1 + r_next) * (c_o ** (-par.gamma)) 

        # Weight by probability and sum
        expected_rhs += prob * rhs_component

    # Multiply by discount factor
    rhs = par.beta * expected_rhs

    # Euler equation error
    err = lhs - rhs
    
    return err

# %%
# Define a function to simulate the transition dynamics of the OLG model
def simulate_olg_ar1(K0, z_series, par: Parameters):
    """
    Simulate the transition dynamics of the OLG model staring for given initial
    capital.

    Parameters
    ----------
    K0 : float
        Initial capital stock.
    z_series : np.ndarray
        Array of TFP values for each period.
    par : Parameters
        Model parameters.

    Returns
    -------
    sim : Simulation
        Simulation object containing the time series for each variable.
    """

    par_sim = copy.copy(par)    # Copy parameters to avoid modifying the original
    T = len(z_series) - 1       # Number of periods
    sim = initialize_sim(T)     # Initialize simulation structure without steady state, hence eq = None
    sim.K[0] = K0               # Set the initial capital stock

    # Iterate over time periods
    for t in range(T + 1):

        # Dynamically choose root-finding interval based on previous period's capital
        low_k = 0.5 * sim.K[t] # Minimum 50% of previous capital or 0.01
        high_k = 2.0 * sim.K[t] # Maximum 200% of previous capital or 0.2

        if t < T:

            # Solve for k_next by finding the root of the Euler error
            res = root_scalar(lambda k_next: euler_err_ar1(k_next, sim.K[t], z_series[t], par_sim),
                bracket=(low_k, high_k),)

            if not res.converged:
                print(f"WARNING: Root-finding failed at t={t}, using fallback value")
                
            # Update capital stock
            sim.K[t + 1] = res.root

        # Update interest rate and wage rate
        sim.r[t], sim.w[t] = compute_prices(sim.K[t], par_sim)

        # Calculate savings rate
        sim.s[t] = compute_savings_rate(sim.r[t], par_sim)

        # Compute pension payments
        p = par_sim.tau * sim.w[t]

        # Compute consumption when young and old
        sim.a[t] = sim.s[t] * sim.w[t] * (1 - par_sim.tau)
        sim.c_y[t] = (1 - par_sim.tau) * sim.w[t] - sim.a[t]
        sim.c_o[t] = (1 + sim.r[t]) * sim.a[t - 1] + p  

        # Update z, tau and output
        sim.z[t] = z_series[t]
        sim.tau[t] = par_sim.tau # Purpose: To keep track of the PAYGO system
        sim.Y[t] = z_series[t] * sim.K[t] ** par_sim.alpha  # Production (L=1)

        # Check goods market clearing
        demand = sim.c_y[t] + sim.c_o[t] + sim.a[t]
        supply = sim.Y[t] + (1 - par_sim.delta) * sim.K[t]
        market_clearing_diff = abs(demand - supply)

        if market_clearing_diff > 1.0e-8:
            print(f"WARNING: Goods market clearing failed at t={t}, with diff={market_clearing_diff:.6e}")

    return sim

# %%
# Simulate the transition dynamics of the OLG model with AR(1) TFP
T = 100         # Number of periods
seed = 1234     # Random seed for reproducibility

# Simulate TFP series
z_series = simulate_ar1(z0, T, seed, par = par_paygo)  # par_paygo = Parameters () where tau = 0.1

# %%
# Simulate OLG model with AR(1) TFP
sim_ar1 = simulate_olg_ar1(eq_pension.K, z_series, par = par_paygo) # par_paygo = Parameters () where tau = 0.1

# %%
# Plot the transition dynamics of the OLG model with AR(1) TFP
plot_simulation(eq_pension, sim_ar1) 

# %%



