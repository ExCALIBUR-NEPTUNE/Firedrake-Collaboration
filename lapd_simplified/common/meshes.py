from firedrake import ExtrudedMesh, mesh, ufl
import numpy as np
from pyop2.mpi import COMM_WORLD
from firedrake.cython import dmcommon


def _DiskMesh(
    radius,
    refinement_level=0,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of the unit disk in 2D

    :kwarg refinement_level: optional number of refinements (0 is a diamond)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    vertices = np.array(
        [
            [0, 0],
            [radius, 0],
            [radius, radius],
            [0, radius],
            [-radius, radius],
            [-radius, 0],
            [-radius, -radius],
            [0, -radius],
            [radius, -radius],
        ],
        dtype=np.double,
    )

    cells = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 7],
            [0, 7, 8],
            [0, 8, 1],
        ],
        np.int32,
    )

    plex = mesh.plex_from_cell_list(
        2, cells, vertices, comm, mesh._generate_default_mesh_topology_name(name)
    )

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        for face in boundary_faces:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
    plex.removeLabel("boundary_faces")
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    coords = plex.getCoordinatesLocal().array.reshape(-1, 2)
    for x in coords:
        norm = np.sqrt(np.dot(x, x))
        if norm > 1.0 / (1 << (refinement_level + 1)):
            t = np.max(np.abs(x)) / norm
            x[:] *= t

    m = mesh.Mesh(
        plex,
        dim=2,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


def CylinderMesh(
    radius, nh, dh, refinement_level=3, longitudinal_axis=2, comm=COMM_WORLD
):
    disk = _DiskMesh(radius, refinement_level=refinement_level, comm=comm)
    cyl = ExtrudedMesh(disk, nh, layer_height=dh / nh, extrusion_type="uniform")

    # Longitudinal axis is 2 after extrusion. Swap coords if necessary

    if longitudinal_axis == 0:
        swap = np.array(cyl.coordinates.dat.data[:, 2])
        cyl.coordinates.dat.data[:, 2] = np.array(cyl.coordinates.dat.data[:, 0])
        cyl.coordinates.dat.data[:, 0] = swap
    elif longitudinal_axis == 1:
        swap = np.array(cyl.coordinates.dat.data[:, 2])
        cyl.coordinates.dat.data[:, 2] = np.array(cyl.coordinates.dat.data[:, 1])
        cyl.coordinates.dat.data[:, 1] = swap
    elif longitudinal_axis == 2:
        pass  # no action required
    else:
        raise ValueError("longitudinal_axis must be 0, 1 or 2")

    return cyl
