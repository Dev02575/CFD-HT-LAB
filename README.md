Overview
This repository contains six numerical simulation experiments covering Computational Fluid Dynamics (CFD), fluid mechanics, heat transfer, and numerical methods.

Each experiment includes a Jupyter Notebook (.ipynb) containing the associated computational work and a PDF report documenting the objectives, methodology, simulation parameters, results, visualizations, and conclusions.

Experiments
Experiment 1: External Flow over a NACA 0012 Airfoil
Objective: To simulate incompressible external flow over a NACA 0012 airfoil and study the effect of angle of attack on velocity, pressure, wake structure, and qualitative force trends.

Reynolds number: 100
Angles of attack: -5°, 0°, 5°, and 10°
Grid: 120 × 60
Total iterations: 300
Files:

Jupyter Notebook
Experiment Report (PDF)
Experiment 2: Fluid Flow and Heat Transfer in a Double-Pipe Heat Exchanger
Objective: To analyze conjugate heat transfer between hot and cold laminar streams separated by a conducting wall.

Cold-stream Reynolds numbers: 50, 100, and 200
Hot-stream Reynolds number: 100
Wall-to-fluid heat-transfer coefficient ratios: 1 and 100
Total iterations: 3000
Files:

Jupyter Notebook
Experiment Report (PDF)
Experiment 3: Internal Flow and Heat Transfer in a Centrifugal Pump
Objective: To study centrifugal pressure rise, velocity components, pump head, and temperature variation in a simplified rotating annular passage.

Reynolds number: 500
Pump speeds: 1000, 2000, and 3000 rpm
Grid: 80 × 160
Total iterations: 1200
Files:

Jupyter Notebook
Experiment Report (PDF)
Experiment 4: Kármán Vortex Shedding over a Circular Cylinder
Objective: To resolve alternating vortex shedding behind a circular cylinder and estimate the Strouhal number.

Reynolds number: 500
Grid: 80 × 160
Total iterations: 1200
Files:

Jupyter Notebook
Experiment Report (PDF)
Experiment 5: Transient Couette Flow — Explicit and Implicit Formulations
Objective: To compare explicit FTCS and implicit backward-Euler solutions of transient viscous momentum diffusion against the analytical series solution.

Kinematic viscosity: 0.01
Upper plate velocity: 1
Grid points: 81
Final simulation time: 2
Explicit Fourier numbers: 0.25, 0.45, and 0.55
Implicit Fourier number: 5
Files:

Jupyter Notebook
Experiment Report (PDF)
Experiment 6: Two-Dimensional Plate Heat Conduction
Objective: To compare Jacobi and line-TDMA/SOR solutions of the steady two-dimensional Laplace equation against an analytical solution.

Fine grid: 81 × 61
Coarse grid: 30 × 15
Source temperature: 100
Maximum iterations: 10000
Convergence tolerance: 1 × 10^-9
Files:

Jupyter Notebook
Experiment Report (PDF)
Numerical Methods and Engineering Topics
The experiments cover:

Incompressible flow and external aerodynamics
Conjugate heat transfer and convection-diffusion
Centrifugal pressure generation and rotating flows
Lattice Boltzmann method and vortex shedding
Explicit and implicit time-integration schemes
Jacobi and line-TDMA/SOR iterative solvers
Numerical stability and convergence analysis
Velocity, pressure, temperature, and vorticity visualization
Tools and Technologies
Python
Jupyter Notebook
Google Colab
Numerical simulation and data visualization
Repository Structure
The repository contains six Jupyter Notebooks and six corresponding PDF experiment reports.

Each notebook is paired with its report through the links provided in the relevant experiment section above.

Author
Dev Kadiya
23BME051

B.Tech Mechanical Engineering
Pandit Deendayal Energy University (PDEU)

Disclaimer
The notebooks and reports document the numerical methods, simulation configurations, and results presented in the respective experiments. Results should be interpreted within the assumptions and limitations of each numerical model.
