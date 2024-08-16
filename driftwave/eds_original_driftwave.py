# Original copy of:
# Nektar-Driftwave_port_June2024.py
# Attempt at time-dependent solver of 2D Hasegawa-Wakatani
# is port of Nektar-Driftwave solver (https://github.com/ExCALIBUR-NEPTUNE/nektar-driftwave/tree/master),
# but fully implicit
# I think this currently only works for single-stage time-steppers
# due to the phi solve being done once per timestep
# TODO - generalize this to higher-order time-steppers
# TODO - can the solver be tuned-up to make it more efficient?
# currently takes c.10 hour for this example to run on laptop on single-core, quite slow
# has a bit of a bug - obvious artifact where the phi value is pinned at the boundary ...
# finally note this script is not described in the m6c5 report, it is a later addition

from firedrake import *
import math
from irksome import Dt, MeshConstant, TimeStepper, RadauIIA
import numpy

# This hack enforces the boundary condition at (0, 0)
class PointwiseBC(DirichletBC):
    @utils.cached_property
    def nodes(self):
        x = self.function_space().mesh().coordinates.dat.data_ro
        zero = numpy.array([0, 10])
        dists = [numpy.linalg.norm(pt - zero) for pt in x]
        minpt = numpy.argmin(dists)
        #print(dists[minpt])
        #if dists[minpt] < 1.0e-6:
        if dists[minpt] < 1.0e-10:
            out = numpy.array([minpt], dtype=numpy.int32)
        else:
            out = numpy.array([], dtype=numpy.int32)
        return out

meshres = 64
L = 40.
mesh = PeriodicSquareMesh(meshres, meshres, L, quadrilateral=True)
V1 = FunctionSpace(mesh, "DG", 3)  # for w (vorticity)
V2 = FunctionSpace(mesh, "DG", 2)  # for n (electron density)
V3 = FunctionSpace(mesh, "CG", 2)  # for phi (electrostatic potential)
V4 = VectorFunctionSpace(mesh, "CG", 2)  # for driftvel (drift velocity)
V = V1*V2

# time parameters (Nektar-Driftwave uses 100k steps of 0.0005)
T = 0.25
# timeres = 2500 # Hard code dt
t = Constant(0.0)
dt = Constant(0.02)

# parameters for irksome
butcher_tableau = RadauIIA(1)  # I think this is backward Euler

# model parameters
alpha = 2.0
kappa = 2.0  # strength of driving force
s = 2.0  # Gaussian width in init data

x, y = SpatialCoordinate(mesh)
wn = Function(V)
w, n = split(wn)
v1, v2 = TestFunctions(V)
phi = TrialFunction(V3)
v3 = TestFunction(V3)

# Gaussian init data
wn.sub(0).interpolate(4.0*((-s*s+(x-20)*(x-20)+(y-20)*(y-20))/(s*s*s*s))*exp(-((x-20)*(x-20)+(y-20)*(y-20))/(s*s)))
wn.sub(1).interpolate(exp(-((x-20)*(x-20)+(y-20)*(y-20))/(s*s)))

# stuff needed to get drift velocity
driftvel = Function(V4)
Lphi = inner(grad(phi),grad(v3))*dx
Rphi = -w*v3*dx
phi_s = Function(V3)
bc1 = DirichletBC(V3, 0, 'on_boundary')
bc_per = PointwiseBC(V3, 0, 'on_boundary')

# TRIALCODE check init data
#File("Nektar-Driftwave_port_June2024_init.pvd").write(wn.sub(0), wn.sub(1), phi_s)
#quit()

norm = FacetNormal(mesh)
driftvel_n = 0.5*(dot(driftvel, norm)+abs(dot(driftvel, norm)))

F = Dt(w)*v1*dx + Dt(n)*v2*dx \
  - v1*div(w*driftvel)*dx - v2*div(n*driftvel)*dx \
  - alpha*(phi_s-n)*(v1+v2)*dx \
  + kappa*grad(phi_s)[1]*v2*dx \
  + driftvel_n('-')*(w('-') - w('+'))*v1('-')*dS \
  + driftvel_n('+')*(w('+') - w('-'))*v1('+')*dS \
  + driftvel_n('-')*(n('-') - n('+'))*v2('-')*dS \
  + driftvel_n('+')*(n('+') - n('-'))*v2('+')*dS \

# params taken from Cahn-Hilliard example:
# https://www.firedrakeproject.org/Irksome/demos/demo_cahnhilliard.py.html
params = {'snes_monitor': None, 'snes_max_it': 100,
          'snes_linesearch_type': 'l2',
          'ksp_type': 'preonly',
          'pc_type': 'lu', 'mat_type': 'aij',
          'pc_factor_mat_solver_type': 'mumps'}

stepper = TimeStepper(F, butcher_tableau, t, dt, wn, solver_parameters=params)

nullspace = VectorSpaceBasis(constant=True)

# this is intended to be direct solver for phi solve
linparams = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu"}

outfile = File("Nektar-Driftwave_port_June2024.pvd")

cnt=0

while float(t) < float(T):
    if (float(t) + float(dt)) >= T:
        dt.assign(T - float(t))
    #solve(Lphi==Rphi, phi_s, solver_parameters=linparams, bcs=bc1)  # breaks at t=4.7 (!)
    #solve(Lphi==Rphi, phi_s, nullspace=nullspace, solver_parameters=linparams, bcs=bc1)
    solve(Lphi==Rphi, phi_s, nullspace=nullspace, solver_parameters=linparams, bcs=bc_per)  # for periodic
    driftvel.interpolate(as_vector([grad(phi_s)[1],-grad(phi_s)[0]]))
    driftvel_n=0.5*(dot(driftvel, norm)+abs(dot(driftvel, norm)))
    if(cnt % 20 == 0):
       print("outputting data ...\n")
       ws, ns = wn.split()
       outfile.write(ws, ns, phi_s)
    cnt=cnt+1
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t), float(dt))

print("done.")
print("\n")
