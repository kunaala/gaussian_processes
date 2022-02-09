#include "Gaussian.hpp"
int main(){
    std::ofstream fd("test.csv");
    Gaussian spgp;
    std::pair<VectorXd,MatrixXd>  res = spgp.training();
    spgp.save_data(res.first);
    Vector<double, Dynamic> s = res.second.diagonal().array().sqrt();
    
    spgp.save_data(s);
  
    
    return 0;
}