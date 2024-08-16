# Driftwave Formulation

$$
\begin{aligned}
\frac{\partial\zeta}{\partial t} + [\phi, \zeta] &= \alpha (\phi - n),\\
\frac{\partial n}{\partial t} + [\phi, n] &= \alpha (\phi - n) - \kappa \frac{\partial\phi}{\partial y},\\
\Delta\phi &= \zeta,
\end{aligned}
$$

where $\zeta$ is the vorticity, $n$ is the perturbed number density and $\phi$ is the electostatic potential. Adiabiacity operator $\alpha$ is a constant and $\kappa$ is the background scale length.
The operator $[a,b]$ is the Poisson bracket defined as

$$
[a,b] = \frac{\partial a}{\partial x}
\frac{\partial b}{\partial y} - 
\frac{\partial a}{\partial y}
\frac{\partial b}{\partial x}.
$$

The domain $\Omega$ is a square with extent $L$ and we apply periodic boundary conditions.
We expect that with these boundary conditions the Poisson solve will only be defined up to a constant.

# Weak Form (Continuous)
We assume that the domain $\Omega$ is discretised into quadrilaterals and that variables $\zeta$, $n$ and $\phi$ are represented by a Continuous Galerkin function space. Expanding the Poisson bracket gives

$$
\begin{aligned}
\frac{\partial\zeta}{\partial t} + 
\frac{\partial \phi}{\partial x}
\frac{\partial \zeta}{\partial y} - 
\frac{\partial \phi}{\partial y}
\frac{\partial \zeta}{\partial x} -
\alpha (\phi - n) &= 0,\\
\frac{\partial n}{\partial t} + 
\frac{\partial \phi}{\partial x}
\frac{\partial n}{\partial y} - 
\frac{\partial \phi}{\partial y}
\frac{\partial n}{\partial x} - 
\alpha (\phi - n) + \kappa \frac{\partial\phi}{\partial y} &= 0,\\
\Delta\phi - \zeta &= 0.
\end{aligned}
$$

Defining

$$
\begin{aligned}
A &= \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}, \\
\hat{\kappa} &= \begin{pmatrix} 0 \\ \kappa \end{pmatrix}.
\end{aligned}
$$

We rewrite the problem as

$$
\begin{aligned}
\frac{\partial\zeta}{\partial t} + 
\nabla \phi \cdot \left(A \nabla \zeta \right) - \alpha (\phi - n) &= 0,\\
\frac{\partial n}{\partial t} + 
\nabla \phi \cdot \left(A \nabla n + \hat{\kappa}\right) - \alpha (\phi - n) &= 0, \\
\Delta \phi - \zeta &= 0.
\end{aligned}
$$

If $\zeta$, $n$ and $\phi$ live in a CG($p$) function space $V$ for some order $p$ then we look for a solution in $V\times V \times V$ with test functions $u$, $v$ and $w$.
Multiplying by the test functions and integrating over the domain gives

$$
\begin{aligned}
 \langle \frac{\partial\zeta}{\partial t}, u \rangle + 
 \langle\nabla \phi \cdot \left(A \nabla \zeta \right), u \rangle - \langle\alpha (\phi - n), u\rangle &= 0,\\
 \langle\frac{\partial n}{\partial t}, v \rangle + 
 \langle\nabla \phi \cdot \left(A \nabla n + \hat{\kappa}\right), v \rangle - \langle \alpha (\phi - n), v \rangle &= 0,\\
 \langle \nabla \phi, \nabla w \rangle + \langle \zeta, w\rangle &= 0.
\end{aligned}
$$

## Weak Form (Nektar++ Discontinuous Formulation)
First we re-write the system as

$$
\begin{aligned}
\frac{\partial\zeta}{\partial t} + 
A^T \nabla \phi \cdot \nabla \zeta - \alpha (\phi - n) &= 0,\\
\frac{\partial n}{\partial t} + 
A^T \nabla \phi \cdot \nabla n + \nabla \phi \cdot \hat{\kappa} - \alpha (\phi - n) &= 0, \\
\Delta \phi - \zeta &= 0.
\end{aligned}
$$

Search for $\zeta$ and $n$ in a CG($p$) function space $V$ and potential $\phi$ in a DG($p$) function space $W$ for some polynomial order $p$.
By multiplying by test functions $u$ and $v$ in $V$ and $w$ in $W$ and integrating over each element we obtain

$$
\begin{aligned}
 \langle \frac{\partial\zeta}{\partial t}, u \rangle + 
 \sum_e \int_{\Gamma_{e}} \left(u \zeta A^T \nabla \phi\right) \cdot \hat{\vec{n}} ~dS - 
 \langle \zeta \nabla \cdot \left( u A^T \nabla \phi \right) \rangle - 
 \langle\alpha (\phi - n), u\rangle &= 0,\\
 \langle\frac{\partial n}{\partial t}, v \rangle + 
 \sum_e \int_{\Gamma_{e}} \left(v n A^T \nabla \phi\right) \cdot \hat{\vec{n}} ~dS - 
 \langle n \nabla \cdot \left( v A^T \nabla \phi \right) \rangle + 
 \langle \nabla \phi \cdot \hat{\kappa}, v \rangle - \langle \alpha (\phi - n), v \rangle &= 0,\\
 \langle \nabla \phi, \nabla w \rangle + \langle \zeta, w\rangle &= 0.
\end{aligned}
$$

where $\hat{\vec{n}}$ is an outward facing unit vector normal to each facet.

To compute the integral over the entire domain $\Omega$ we sum the contributions from the elements.
When the sum is computed over all elements each internal edge makes two contributions, denoted $+$ and $-$, from the two elements which share a facet,

$$
\begin{aligned}
\sum_ e \int_{\Gamma_{e}} F \cdot \hat{\vec{n}} ~dS &= 
\int_{\Gamma_{\text{internal}}} F_+\cdot \hat{\vec{n}}_ + + F_-\cdot \hat{\vec{n}}_ - ~dS.
\end{aligned}
$$

Following the standard upwind approach, for a vector "velocity" $\vec{u}$ we define

$$
    \hat{u} := \frac{1}{2} \left( \vec{u} \cdot \hat{\vec{n}} + |\vec{u} \cdot \hat{\vec{n}}|\right)
$$

which takes the value of $\vec{u} \cdot \hat{\vec{n}}$ when $\vec{u} \cdot \hat{\vec{n}} > 0$ and 0 otherwise.
If

$$
\hat{\phi} = \frac{1}{2} \left( A^T \nabla \phi \cdot \hat{\vec{n}} + \left| A^T \nabla \phi \cdot \hat{\vec{n}}\right|\right)
$$

then we can write

$$
\begin{aligned}
\sum_{e} \int_{\Gamma_{e}} \left(u \zeta A^T \nabla \phi\right) \cdot \hat{\vec{n}} ~dS \approx
\int_{\Gamma_{\text{internal}}}&
\left(u_+ - u_-\right) \left(\zeta_+ \hat{\phi}_ + - \zeta_- \hat{\phi}_ - \right) ~dS
\\
\end{aligned}
$$

and a similar expression for the density equation by replacing $\zeta$ with $n$ and $u$ for $v$.

