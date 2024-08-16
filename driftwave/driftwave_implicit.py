from firedrake import *
from petsc4py import PETSc

meshres = 64
L = 40
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = PeriodicSquareMesh(meshres, meshres, L, quadrilateral=True,
                          distribution_parameters=distribution_parameters)

# time parameters (Nektar-Driftwave uses 100k steps of 0.0005)
T = 50.
dt = 0.1
dtc = Constant(dt)
dumprate = 10

deg = 3
R = FunctionSpace(mesh, "R", 0)
DGk = FunctionSpace(mesh, 'DG', deg)
CGk = FunctionSpace(mesh, 'CG', deg)

W = DGk*DGk*CGk

U0 = Function(W)
U1 = Function(W)

# some model parameters
alpha = Constant(2.0, domain=mesh)
kappa = Constant(2.0, domain=mesh)

# split variables for assigning ICs
zeta0, n0, phi0 = U0.subfunctions

# IC parameters
s = Constant(2.0)

# ICs
x, y = SpatialCoordinate(mesh)
zeta0.interpolate(4.0*((-s*s+(x-20)*(x-20)+(y-20)*(y-20))/(s*s*s*s))*exp(-((x-20)*(x-20)+(y-20)*(y-20))/(s*s)))
n0.interpolate(exp(-((x-20)*(x-20)+(y-20)*(y-20))/(s*s)))

# shift n0 and zeta0
# for periodic nullspace
One = Constant(1.0, domain=mesh)
ddv = TestFunction(DGk)
nbar = assemble(n0*dx)/assemble(One*dx)
n0 -= nbar
zetabar = assemble(zeta0*dx)/assemble(One*dx)
zeta0 -= zetabar

# split variables for writing forms
zeta0, n0, phi0 = split(U0)
zeta1, n1, phi1 = split(U1)
# split variables for writing formszeta1, n1, phi1 = split(U1)
dzeta, dn, dphi = TestFunctions(W)

zetah = (zeta0 + zeta1)/2
nh = (n0 + n1)/2
phih = phi1 #

uh = as_vector([-phih.dx(1), phih.dx(0)])

normal = FacetNormal(mesh)
un = 0.5*(dot(uh, normal) + abs(dot(uh, normal)))

def make_eqn(zeta0, zeta1, zetah, dzeta):
    eqn = inner(zeta1 - zeta0, dzeta)*dx
    eqn += dtc*(
        -inner(grad(dzeta), zetah*uh)*dx
        + (dzeta('+') - dzeta('-'))*(un('+')*zetah('+')
                                     - un('-')*zetah('-'))*dS
    )
    eqn -= dtc*alpha*(phih - nh)*dzeta*dx
    return eqn

eqn = make_eqn(zeta0, zeta1, zetah, dzeta)
eqn += make_eqn(n0, n1, nh, dn) + dtc*dn*kappa*phih.dx(1)*dx
eqn += inner(grad(phih), grad(dphi))*dx + zetah*dphi*dx

peqn = eqn + phih*dphi*dx
Jp = derivative(peqn, U1)

class RPC(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        a = test*trial*dx
        return (a, None)

pparams = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "snes_atol": 1.0e-50,
    "snes_stol": 1.0e-50,
    "snes_rtol": 1.0e-7,
    "ksp_atol": 1.0e-50,
    "ksp_rtol": 1.0e-9,
    "ksp_max_it": 50,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "bjacobi",
    "fieldsplit_0_sub_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_ksp_reuse_preconditioner": None,
    "fieldsplit_1_pc_type": "lu",
    "fieldsplit_1_pc_factor_mat_solver_type": "mumps"
}

dw_prob = NonlinearVariationalProblem(eqn, U1, Jp=Jp)
dw_solver = NonlinearVariationalSolver(dw_prob,
                                       solver_parameters=pparams)

outfile = File("driftwave.pvd")

t = 0.

# split variables for writing data
zeta0, n0, phi0 = U0.subfunctions
zeta_out = Function(DGk)
n_out = Function(DGk)
zeta_out.assign(zeta0)
n_out.assign(n0)
outfile.write(zeta_out, n_out)

U1.assign(U0)

dumpclock = 0
first = True
while t < T - dt/2:
    PETSc.Sys.Print(t)
    t += dt
    dw_solver.solve()
    U0.assign(U1)

    # periodic stuff
    phibar = assemble(phi0*dx)/assemble(One*dx)
    phi0 -= phibar
    
    dumpclock += 1
    if dumpclock == dumprate:
        zeta_out.assign(zeta0)
        n_out.assign(n0)
        outfile.write(zeta_out, n_out)
        dumpclock = 0
