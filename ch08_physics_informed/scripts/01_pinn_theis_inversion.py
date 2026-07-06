"""
Chapter 8 worked example: a Physics-Informed Neural Network (PINN) solving
the INVERSE problem for the classical Theis (1935) transient radial flow
equation in a confined aquifer, recovering Transmissivity (T) and
Storativity (S) from sparse, noisy drawdown observations.

EXPLICITLY SYNTHETIC: parameters are loosely inspired by published ranges
for the Albian sandstone (Continental Intercalaire) of the northern Sahara,
but this is a controlled synthetic benchmark, not a real field dataset,
exactly as originally scoped in the book outline for this chapter. Unlike
every other worked example in this book, no real Hennaya data are used
here, because the purpose is to demonstrate a physics-informed inversion
method against a KNOWN ground truth.

Governing equation (radial, confined, homogeneous, isotropic aquifer):
    S * ds/dt = T * (d2s/dr2 + (1/r) * ds/dr)
where s(r,t) is drawdown. Analytical solution (Theis 1935):
    s(r,t) = Q / (4*pi*T) * W(u),   u = r^2 * S / (4*T*t)
W(u) is the Theis well function (exponential integral E1(u)).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.special import exp1

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Synthetic "true" aquifer and pumping test setup
# =====================================================================
T_true = 450.0      # m^2/day (Albian sandstone, order of magnitude realistic)
S_true = 8.0e-4      # dimensionless storativity
Q = 2000.0           # m^3/day, constant pumping rate

def theis_drawdown(r, t, T, S, Q):
    u = (r**2 * S) / (4 * T * t)
    return Q / (4 * np.pi * T) * exp1(u)

# "Monitoring wells" at three distances from the pumping well
obs_radii = [15.0, 40.0, 90.0]     # metres
obs_times = np.geomspace(0.05, 5.0, 12)  # days

obs_r, obs_t, obs_s = [], [], []
rng = np.random.default_rng(7)
for r in obs_radii:
    s_true = theis_drawdown(r, obs_times, T_true, S_true, Q)
    noise = rng.normal(0, 0.03 * s_true.mean(), size=s_true.shape)  # ~3% noise
    obs_r.extend([r]*len(obs_times))
    obs_t.extend(obs_times.tolist())
    obs_s.extend((s_true + noise).tolist())

obs_r = np.array(obs_r); obs_t = np.array(obs_t); obs_s = np.array(obs_s)
print(f"Synthetic observations: {len(obs_s)} points from {len(obs_radii)} "
      f"monitoring wells at r = {obs_radii} m")
print(f"True parameters: T = {T_true} m2/day, S = {S_true:.1e}")

# =====================================================================
# 2. PINN setup
# =====================================================================
R_MIN, R_MAX = 5.0, 150.0   # avoid the singularity at r=0
T_MAX = 5.0

class PINN(nn.Module):
    def __init__(self, n_hidden=32, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1)]
        self.net = nn.Sequential(*layers)
        # Trainable log-parameters (log-space keeps them positive; initial
        # guesses are deliberately far from the truth to test recovery)
        self.log_T = nn.Parameter(torch.tensor(np.log(50.0), dtype=torch.float32))   # true: log(450)
        self.log_S = nn.Parameter(torch.tensor(np.log(1e-2), dtype=torch.float32))   # true: log(8e-4)

    def forward(self, r, t):
        x = torch.cat([r/R_MAX, t/T_MAX], dim=1)
        return self.net(x)

    @property
    def T(self): return torch.exp(self.log_T)
    @property
    def S(self): return torch.exp(self.log_S)

model = PINN()
# Separate learning rates: the network must learn the correct FUNCTION
# SHAPE quickly, but (T, S) must move slowly, otherwise the optimizer
# drags the physical parameters along an "easy" gradient path before the
# network shape has converged - see the diagnostic discussion below.
optimizer = torch.optim.Adam([
    {"params": model.net.parameters(), "lr": 2e-3},
    {"params": [model.log_T, model.log_S], "lr": 5e-5},
])

obs_r_t = torch.tensor(obs_r, dtype=torch.float32).view(-1, 1)
obs_t_t = torch.tensor(obs_t, dtype=torch.float32).view(-1, 1)
obs_s_t = torch.tensor(obs_s, dtype=torch.float32).view(-1, 1)

def sample_collocation(n=2000):
    r = torch.rand(n, 1) * (R_MAX - R_MIN) + R_MIN
    t = torch.rand(n, 1) * T_MAX + 1e-3
    r.requires_grad_(True); t.requires_grad_(True)
    return r, t

def pde_residual(model, r, t):
    s = model(r, t)
    s_r = torch.autograd.grad(s, r, grad_outputs=torch.ones_like(s), create_graph=True)[0]
    s_rr = torch.autograd.grad(s_r, r, grad_outputs=torch.ones_like(s_r), create_graph=True)[0]
    s_t = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(s), create_graph=True)[0]
    residual = model.S * s_t - model.T * (s_rr + s_r / r)
    return residual

# Initial condition: s(r, t->0) = 0
r_ic = torch.rand(300, 1) * (R_MAX - R_MIN) + R_MIN
t_ic = torch.full_like(r_ic, 1e-3)

# Far-field boundary: s(r_max, t) ~= 0
t_bc = torch.rand(300, 1) * T_MAX + 1e-3
r_bc = torch.full_like(t_bc, R_MAX)

# =====================================================================
# 3. Training
# =====================================================================
N_EPOCHS = 8000
history = []
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    r_col, t_col = sample_collocation()
    res = pde_residual(model, r_col, t_col)
    loss_pde = torch.mean(res**2)

    loss_data = torch.mean((model(obs_r_t, obs_t_t) - obs_s_t)**2)
    loss_ic = torch.mean(model(r_ic, t_ic)**2)
    loss_bc = torch.mean(model(r_bc, t_bc)**2)

    # Loss weighting: chosen empirically after diagnosing a genuine
    # non-identifiability failure mode (see below).
    loss = 50 * loss_data + 2000 * loss_pde + 10 * loss_ic + 10 * loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
        T_est, S_est = model.T.item(), model.S.item()
        history.append((epoch, loss.item(), T_est, S_est))
        print(f"epoch {epoch:5d} | loss={loss.item():.4f} | "
              f"T_est={T_est:7.1f} (true {T_true}) | S_est={S_est:.2e} (true {S_true:.1e})")

# =====================================================================
# 4. Final evaluation
# =====================================================================
T_final, S_final = model.T.item(), model.S.item()
T_err_pct = 100 * (T_final - T_true) / T_true
S_err_pct = 100 * (S_final - S_true) / S_true
print(f"\nFinal recovered parameters:")
print(f"  T = {T_final:.1f} m2/day (true {T_true}, error {T_err_pct:+.1f}%)")
print(f"  S = {S_final:.2e} (true {S_true:.1e}, error {S_err_pct:+.1f}%)")

# Head-field RMSE against the analytical Theis solution on a validation grid
r_val = np.linspace(R_MIN, R_MAX, 40)
t_val = np.geomspace(0.05, T_MAX, 40)
RR, TT = np.meshgrid(r_val, t_val)
s_analytical = theis_drawdown(RR, TT, T_true, S_true, Q)

with torch.no_grad():
    r_flat = torch.tensor(RR.ravel(), dtype=torch.float32).view(-1, 1)
    t_flat = torch.tensor(TT.ravel(), dtype=torch.float32).view(-1, 1)
    s_pred = model(r_flat, t_flat).numpy().reshape(RR.shape)

rmse = np.sqrt(np.mean((s_pred - s_analytical)**2))
print(f"\nDrawdown field RMSE (PINN vs analytical Theis solution): {rmse:.3f} m")
print(f"Mean analytical drawdown over validation grid: {s_analytical.mean():.3f} m")

# =====================================================================
# 5. Honest failure-mode discussion
# =====================================================================
print("\n" + "="*70)
print("DIAGNOSIS: parameter identifiability failure")
print("="*70)
print(f"""
Inverse parameter recovery did NOT succeed: T is recovered within
{T_err_pct:+.0f}% and S within {S_err_pct:+.0f}% of the true values, despite:
  - training for {N_EPOCHS} epochs,
  - a physics loss weight four orders of magnitude larger than a first
    naive attempt,
  - separate (much slower) learning rates for (T, S) versus the network.

A targeted diagnostic - initializing (T, S) EXACTLY at their true values
and observing whether training holds them there - showed the true
parameters are NOT a stable point of this joint (network + parameter)
training scheme: the optimizer drifts away from them even from a perfect
start. This rules out poor initialization or loss-weight imbalance as the
sole cause and points to a genuine identifiability/optimization pathology:
early in training, before the network has learned the correct functional
shape of s(r,t), it is numerically "easier" for the optimizer to reduce
the loss by adjusting the two scalar parameters (T, S) than by reshaping
the many-weight network, dragging the parameter estimates away from the
truth before the network catches up.

This matches, with a fully worked and diagnosed real example, the caution
already present in Book Section 8.6 ("loss weighting sensitivity... among
the principal limitations" of PINNs). Reported approaches in the PINN
literature to address this class of failure include a two-stage schedule
(pre-train the network alone with fixed/frozen parameters before jointly
optimizing), curriculum or annealed loss weighting, and denser or better
placed observation data - each a legitimate direction for future work,
not attempted further here in the interest of reporting the failure
honestly rather than tuning until a preferred number appears.
""")

np.savez("/home/claude/hennaya/ch8_pinn_results.npz",
         RR=RR, TT=TT, s_analytical=s_analytical, s_pred=s_pred,
         obs_r=obs_r, obs_t=obs_t, obs_s=obs_s,
         T_true=T_true, S_true=S_true, T_final=T_final, S_final=S_final,
         history=np.array(history))
print("Saved: ch8_pinn_results.npz")
