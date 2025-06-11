{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    python3

    python3Packages.numpy
    python3Packages.pandas
    python3Packages.seaborn
    python3Packages.scipy
    python3Packages.statsmodels
    python3Packages.matplotlib
    python3Packages.scikit-learn
    python3Packages.xgboost
    python3Packages.shap
  ];
}