#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Compare platforms using different tolerances.

DESCRIPTION

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

TODO

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import sys
import math
import copy
import time

import numpy

# OpenMM imports
import simtk.unit as units
import simtk.openmm as openmm
from simtk.openmm import app
    
import netCDF4 as netcdf # for netcdf interface provided by netCDF4 in enthought python

#=============================================================================================
# INTEGRATORS
#=============================================================================================

def VelocityVerletIntegrator(timestep):
    """
    Construct a velocity Verlet integrator.

    ARGUMENTS

    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep

    RETURNS

    integrator (simtk.openmm.CustomIntegrator) - a velocity Verlet integrator

    NOTES

    This code is verbatim from Peter Eastman's example.

    """
    
    integrator = openmm.CustomIntegrator(timestep)

    integrator.addPerDofVariable("x1", 0)

    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    integrator.addConstrainVelocities()
    
    return integrator

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def create_nacl_system():
    # Read ion pair.
    pdbfile = app.PDBFile('nacl.pdb')

    # Load forcefield.
    forcefields_to_use = ['tip3p.xml', 'amber96.xml']
    forcefield = app.ForceField(*forcefields_to_use)

    # Solvate.
    modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=10.0*units.angstroms)

    # Delete any added ions.
    delete_list = list()
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            if residue.name == "CL":
                delete_list.append(residue)
    print delete_list
    print "Deleting %d residues..." % len(delete_list)
    modeller.delete(delete_list)

    # Create System.
    cutoff = 9.0 * units.angstroms
    #nonbondedMethod = app.CutoffPeriodic
    nonbondedMethod = app.PME
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=cutoff, rigidWater=True)

    # Extract positions.
    positions = modeller.positions
    
    # Extract topology.
    topology = modeller.topology

    return [system, positions, topology]

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":

    temperature = 298.0 * units.kelvin
    collision_rate = 20.0 / units.picoseconds
    timestep = 2.0 * units.femtoseconds
    pressure = 1.0 * units.atmospheres

    niterations = 1000 # number of iterations
    nsteps = 500 # number of timesteps per iteration
    
    store_filename = 'nacl.nc'

    # Specify platform options.
    #options = {"CudaPrecision": "double"} 
    #options = {"CudaPrecision": "mixed"} 
    #tol = 1.0e-8 # constraint tolerance
    platform_name = "CUDA"

    platform = openmm.Platform.getPlatformByName(platform_name)

    #
    # Set up system.
    #

    print "Creating solvated NaCl system..."

    [system, positions, topology] = create_nacl_system()
    print "system has %d atoms" % system.getNumParticles()
    
    # DEBUG
    outfile = open('nacl-solvated.pdb', 'w')
    app.PDBFile.writeFile(topology, positions, file=outfile)
    outfile.close()

    #
    # Create alchemical intermediates.
    #

    print "Creating alchemical intermediates..."

    # Create a factory to produce alchemical intermediates.
    ligand_atoms = range(0,2) # NaCl
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    # Get the default protocol for 'denihilating' in complex in explicit solvent.
    protocol = factory.defaultComplexProtocolImplicit()
    # Create the perturbed systems using this protocol.
    systems = factory.createPerturbedSystems(protocol)

    #
    # Set up replica exchange simulation.
    #

    print "Setting up replica-exchange simulation..."

    # Create reference state.
    from thermodynamics import ThermodynamicState
    reference_state = ThermodynamicState(reference_system, temperature=temperature, pressure=pressure)
    # Create simulation.
    simulation = HamiltonianExchange(reference_state, systems, coordinates, store_filename)
    simulation.number_of_iterations = niterations # set the simulation to only run 2 iterations
    simulation.timestep = timestep # set the timestep for integration
    simulation.nsteps_per_iteration = nsteps # run 50 timesteps per iteration
    simulation.minimize = False
    simulation.platform = platform # set platform

    # Run simulation.
    simulation.run() # run the simulation

    
