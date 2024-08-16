from firedrake import *
from petsc4py import PETSc

meshres = 64
L = 40
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = PeriodicSquareMesh(meshres, meshres, L, quadrilateral=True,
                          distribution_parameters=distribution_parameters)

# time parameters (Nektar-Driftwave uses 100k steps of 0.0005)
T = 50.
dt = 0.01
dtc = Constant(dt)
dumprate = 100

deg = 3
DGk = FunctionSpace(mesh, 'DG', deg)
CGk = FunctionSpace(mesh, 'CG', deg)

n0 = Function(DGk, name="n")
zeta0 = Function(DGk, name="zeta")
phi0 = Function(CGk, name="phi")

dn1 = Function(DGk)
n1 = Function(DGk)
dzeta1 = Function(DGk)
zeta1 = Function(DGk)

# some model parameters
alpha = Constant(2.0, domain=mesh)
kappa = Constant(2.0, domain=mesh)

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

phi = TrialFunction(CGk)
psi = TestFunction(CGk)

# Build the weak form for the inversion
Aphi = inner(grad(psi), grad(phi)) * dx
Lphi = -zeta1 * psi * dx

phiparams = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'}

phi_problem = LinearVariationalProblem(Aphi, Lphi, phi0)
v_basis = VectorSpaceBasis(constant=True)
phi_solver = LinearVariationalSolver(
    phi_problem, solver_parameters=phiparams,
    nullspace = v_basis)


gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))
uh = as_vector([-phi0.dx(1), phi0.dx(0)])
normal = FacetNormal(mesh)
un = 0.5 * (dot(gradperp(phi0), normal) + abs(dot(gradperp(phi0), normal)))

q = TrialFunction(DGk)
p = TestFunction(DGk)
a_mass = p * q * dx

ops = (
    -inner(grad(p), q*gradperp(phi0))*dx
    + (p('+') - p('-'))*(un('+')*q('+')
                                 - un('-')*q('-'))*dS
)
ops += alpha*(phi0 - n1)*p*dx

rhs = a_mass - dtc * ops
dispersion =  dtc*p*kappa*phi0.dx(1)*dx
nop = action(rhs, n1)
zetaop = action(rhs, zeta1) - dispersion

nproblem = LinearVariationalProblem(a_mass, nop, dn1)
zetaproblem = LinearVariationalProblem(a_mass, zetaop, dzeta1)

n_solver = LinearVariationalSolver(nproblem,
                                   solver_parameters={"ksp_type": "preonly",
                                                      "pc_type": "bjacobi",
                                                      "sub_pc_type": "ilu"})

zeta_solver = LinearVariationalSolver(zetaproblem,
                                      solver_parameters={"ksp_type": "preonly",
                                                         "pc_type": "bjacobi",
                                                         "sub_pc_type": "ilu"})



outfile = File("driftwave.pvd")

t = 0.

def both(e):
    return 2*avg(e)

# Courant number bits
DG0 = FunctionSpace(mesh, "DG", 0)
One = Function(DG0).assign(1.0)
n = FacetNormal(mesh)
zeta1.assign(zeta0)
phi_solver.solve()
unn = 0.5*(inner(-uh, n) + abs(inner(-uh, n))) # gives fluxes *into* cell only
vdg = TestFunction(DG0)
Courant_num = Function(DG0, name="Courant numerator")
Courant_num_form = dtc*both(unn*vdg)*dS
Courant_denom = Function(DG0, name="Courant denominator")
assemble(One*vdg*dx, tensor=Courant_denom)
Courant = Function(DG0, name="Courant")

assemble(Courant_num_form, tensor=Courant_num)
Courant.interpolate(Courant_num/Courant_denom)

outfile.write(n0, zeta0, phi0, Courant)

dumpclock = 0
while t < T - dt/2:
    PETSc.Sys.Print(t)
    t += dt

    # SSPRK3 stage 1
    zeta1.assign(zeta0)
    n1.assign(n0)
    phi_solver.solve()
    zeta_solver.solve()
    n_solver.solve()

    # SSPRK3 stage 2
    zeta1.assign(dzeta1)
    n1.assign(dn1)
    phi_solver.solve()
    zeta_solver.solve()
    n_solver.solve()

    # SSPRK3 stage 3
    zeta1.assign(0.75 * zeta0 + 0.25 * dzeta1)
    n1.assign(0.75 * n0 + 0.25 * dn1)
    phi_solver.solve()
    zeta_solver.solve()
    n_solver.solve()

    # SSPRK3 update
    zeta0.assign(zeta0 / 3 + 2 * dzeta1 / 3)
    n0.assign(n0 / 3 + 2 * dn1 / 3)
    
    dumpclock += 1
    if dumpclock == dumprate:
        assemble(Courant_num_form, tensor=Courant_num)
        Courant.interpolate(Courant_num/Courant_denom)

        outfile.write(n0, zeta0, phi0, Courant)
        dumpclock = 0
