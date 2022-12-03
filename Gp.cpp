#include<iostream>
#include<random>
#include<Eigen/Dense>
#include <fstream>
#include <cmath>



Eigen::MatrixXd kernel(Eigen::MatrixXd X1,Eigen::MatrixXd X2, 
                        unsigned int sigma_f=1,unsigned int l=1 ){


    Eigen::MatrixXd M =X1.cwiseAbs2().rowwise().sum().col(0).replicate(1,X2.rows());
    Eigen::RowVectorXd N = X2.cwiseAbs2().rowwise().sum().col(0);
    

    // std::cout<<"Msize\n"<<M.rows()<<"x"<<M.cols()<<'\n';
    

    // std::cout<<"M+Nsize\n"<<(M.rowwise()+N.transpose()).rows()<<"x"<<(M.rowwise()+N.transpose()).cols()<<'\n';

    return  pow(sigma_f,2) * ( (M.rowwise()+N - 2 * X1*X2.transpose())  / -2*pow(l,2) ).array().exp();                                
}

 Eigen::MatrixXd  gen_samples(std::pair<int,int> limit,const int dim_, const int size_){
    
    std::vector<std::default_random_engine> gen(dim_);
    std::uniform_real_distribution<double> unigen(limit.first,limit.second);
    Eigen::MatrixXd X_s(size_,dim_);
    for(unsigned int i=0;i<dim_;i++){
        gen.at(i).seed(i+1);
        for(unsigned int j=0;j<size_;j++)  X_s(j,i) = unigen(gen.at(i));
    }
    return X_s; 

}

void save_data(Eigen::MatrixXd M, std::string filename ){

    /*
     * Saves Eigen Vector "v" in the file specified in "filename" in CSV format.
     */
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, " ");

	std::ofstream fd(filename,std::ios::app);
	if (fd.is_open()){
        fd << M.format(CSVFormat);
        fd <<'\n';
		fd.close();
	}
}

std::pair<Eigen::MatrixXd,Eigen::MatrixXd> posterior(Eigen::MatrixXd X_train, Eigen::MatrixXd F_train, Eigen::MatrixXd X_test){

    Eigen::MatrixXd K = kernel(X_train,X_train);
    Eigen::MatrixXd K_inv = K.inverse();
    Eigen::MatrixXd K_t = kernel(X_train,X_test);
    Eigen::MatrixXd K_tt = kernel(X_test,X_test);

    Eigen::MatrixXd mu_t = K_t.transpose() *K_inv * F_train;
    // std::cout<<"mu_size\n"<<mu_t.rows()<<"x"<<mu_t.cols()<<'\n';

    Eigen::MatrixXd covar_t = K_tt - K_t.transpose() * K_inv * K_t;

    return std::make_pair(mu_t,covar_t);

}  

// Eigen::MatrixXd predict(std::pair<Eigen::MatrixXd,Eigen::MatrixXd> gp_params,Eigen::MatrixXd X){
//     unsigned int d = X.cols();
//     double norm_factor = (pow(2*M_PI,-d/2.) * pow(gp_params.second.determinant(),-0.5));
//     std::cout<<norm_factor<<'\n';
//     return ((X - gp_params.first).transpose() * gp_params.second.inverse() * (X - gp_params.first)/-2).array().exp() * norm_factor;
// }

int main(){
    
    const unsigned int dim_ = 1;
    const unsigned int train_size_ = 200;
    const unsigned int test_size_ = 50;
    std::pair<int,int> limit(-5,5);
    std::string fname="plot.csv";
    std::ofstream fd(fname,std::ios::out);
    if (fd.is_open()) fd<<dim_<<'\n';

    
    Eigen::MatrixXd X_train = gen_samples(limit,dim_,train_size_);
    Eigen::MatrixXd F_train;
    /**
    * COS/SINE
    */
    if(dim_==2) F_train = X_train.cwiseAbs2().rowwise().sum().array().cos();
    else F_train = X_train.array().cos();
    /**
     * @brief Sphere
     * 
     */
    // F_train = X_train.cwiseAbs2().rowwise().sum().cwiseSqrt();
    Eigen::Matrix<double,test_size_,dim_> X_test(gen_samples(limit,dim_,test_size_));

    save_data(X_train.transpose(),"plot.csv");
    save_data(F_train.transpose(),"plot.csv");
    save_data(X_test.transpose(),"plot.csv");
    
    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> gp_params;
    // gp_params.first = Eigen::MatrixXd::Zero(test_size_,dim_);
    // gp_params.second = kernel(X_test,X_test);
   

    gp_params = posterior(X_train,F_train,X_test);
    save_data(gp_params.first.transpose(),"plot.csv");
    save_data(gp_params.second.diagonal().transpose(),"plot.csv");
    


    
    return 0;
}