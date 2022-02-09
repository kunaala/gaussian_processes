#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include "iniparser.hpp"

using namespace Eigen;

class Gaussian{
    public:
        Gaussian();
        std::pair<VectorXd,MatrixXd>  training();
        void save_data(Vector<double, Dynamic> v, std::string filename = "test.csv");

        Vector<double, Dynamic> mu_pred;
        Matrix<double, Dynamic, Dynamic> X, Xp, Xt,y,sigma_pred;
        double noise_var=0.;

    
    private:
        void gen_data();
        Matrix<double, Dynamic, Dynamic>  kernel_fn(Matrix<double, Dynamic, Dynamic> X1, Matrix<double, Dynamic,Dynamic> X2);
        Vector<double, Dynamic> func_gen(Vector<double, Dynamic> x, float noise_var=0.);
        Vector<double, Dynamic> linspace(unsigned int size, float low, float high);
        Vector<double, Dynamic> unigen(unsigned int size, float low, float high);
        float ini_param(const char* key, const char* sector = "data", std::string filename = "gaussian.ini");



};
#endif // GAUSSIAN_H