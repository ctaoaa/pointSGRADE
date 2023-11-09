# PointSGRADE: Sparse Learning with Graph Representation for Anomaly Detection by Using Unstructured 3D Point Cloud Data

Accepted by *IISE Transactions*

## Requirements

Mandatory: cvxpy, numba, numpy+mkl, open3d, scikit-learn, sparse-dot-mkl, tifffile, PIL, bspline

Optional: cvxopt, matplotlib-label-lines *(only for compare_regularizers.py)*

## Usages

### In *utils* folder:

*convex_optimization.py*:  formulation with group lasso penalty, used for initialization

*generate_anomaly_random_sample.py*: generate a simulated point cloud with 10 pinholes with random depths, radii, and directions

*generate_anomaly_sample.py*: generated a simulated point cloud with pre-defined parameters

*initialization_funcs.py*: functions for initialization

*my_func.py*: functions

*pointSGRADE_solution_mkl_float32.py*: main anomaly detection algorithm

    Main function: pointSGRADE_solver(input point cloud, hyperparameters)
    Return: estimated anomaly label and other information

### In main branch:

*expr_example.py*: results of the representative sample in the manuscript

*expr_great_variety.py*: results under the great variety of anomalies in the manuscript

*expr_real_sample.py*: results of the real-world sample (data in *mvtec* folder) in the manuscript

*expr_sensitivity_noise.py*: results of sensitivity of noise in the supplementary material

*expr_sensitivity_depth.py*: results of sensitivity of depth (pinhole) in the supplementary material

*expr_sensitivity_radius.py*: results of sensitivity of radius (pinhole) in the supplementary material

*expr_sensitivity_neighborhood_size.py*: results of sensitivity of neighborhood size (hyperparameter k) in the supplementary material

*compare_regularizers.py*: stylized example to illustrate the necessity of a non-convex penalty in the supplementary material

*analysis_\*.py*: printing or plotting related results

## Contact

For any question, feel free to contact me through the email *ctaoaa@connect.ust.hk*







