#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "force_functions.h"
#include "simulation.h"

namespace py = pybind11;

PYBIND11_MODULE(molsim, m) {
    m.doc() = "A C++ accelerated module for Lennard-Jones molecular simulations";

    m.def("forces", &forces, "Compute 3D Lennard-Jones forces and potential",
          py::arg("coordinates"), py::arg("L"), py::arg("N"), py::arg("r_c"));

    m.def("force2d", &force2d, "Compute 2D Lennard-Jones forces and potential",
          py::arg("coordinates"), py::arg("L"), py::arg("N"), py::arg("r_c"));

    // By convention, Python class names are in PascalCase (e.g., Simulation)
    py::class_<simulation>(m, "Simulation")
        .def(py::init<double, double, int>(),
             py::arg("length"), py::arg("cutoff"), py::arg("num_particles"))
        .def("makegrid", &simulation::makegrid, "Build the neighbor list grid",
             py::arg("coordinates"))
        .def("force2dhp", &simulation::force2dhp, "Compute 2D forces using the grid",
             py::arg("coordinates"));
}