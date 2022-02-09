#include "Gaussian.hpp"

Gaussian::Gaussian(){
    Gaussian::gen_data();
}

float Gaussian::ini_param(const char* key, const char* sector, std::string filename){
    /*
     *Loads parameters from the .ini file
     */
    INI::File param;
    if (!param.Load(filename)){
        std::cout<<"error opening file"<<'\n';
	}
    // Load kernel Hyperparameters
    return param.GetSection(sector)->GetValue(key).AsDouble();

}

void Gaussian::save_data(Vector<double, Dynamic> v, std::string filename ){

    /*
     * Saves Eigen Vector "v" in the file specified in "filename" in CSV format.
     */
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ");

	std::ofstream fd(filename,std::ios::app);
	if (fd.is_open()){
		fd << v.transpose().format(CSVFormat);
        fd <<'\n';
		fd.close();
	}
}

Vector<double, Dynamic> Gaussian::unigen(unsigned int size, float low, float high){
    /*
     *Generating uniformly distributed real "size" numbers over a specified interval (low,high).
     */
    Vector<double, Dynamic> v(size);
    std::default_random_engine gen;
	std::uniform_real_distribution<double> d(low,high);
    for(int i=0; i<size;i++){
		v(i) = d(gen);
    }
    return v;

}

Vector<double, Dynamic> Gaussian::linspace(unsigned int size, float low, float high){
    /*
     *Generating evenly spaced "size" numbers over a specified interval (low,high).
     */
    Vector<double, Dynamic> v(size);
    float step = (high - low) /size;
	for(int i=0; i<size;i++){
		v(i) = low + i * step;  
	} 
    return v;   

}



Vector<double, Dynamic> Gaussian::func_gen(Vector<double, Dynamic> x, float noise_var){
     /*
     *Generates y = f(x) + noise_var * noise.
     */
    // Vector<double, Dynamic> y = x.array().sin() + noise_var * (rangen(x.size(), -1.0,1.0)).array();
    std::default_random_engine gen;
    std::normal_distribution<> rand{0, noise_var};
    Vector<double, Dynamic> noise(x.rows());
    for(int i=0; i<x.rows();i++){
		noise(i) = rand(gen);  
	}
    Vector<double, Dynamic> y = x.array().sin() + noise_var * noise.array();

    return y;
}

Matrix<double, Dynamic, Dynamic>  Gaussian::kernel_fn(Matrix<double, Dynamic, Dynamic> X1, Matrix<double, Dynamic,Dynamic> X2){
    float kernel_var = Gaussian::ini_param("var","kernel"), kernel_length = Gaussian::ini_param("l", "kernel");
    Matrix<double, Dynamic, Dynamic> sq_dist;
    sq_dist  = X1.rowwise().squaredNorm() * VectorXd::Ones(X2.rows()).transpose()
                + VectorXd::Ones(X1.rows()) * X2.rowwise().squaredNorm().transpose()
                - 2 * X1 * X2.transpose();
    return kernel_var * (-0.5 * sq_dist / pow(kernel_length,2)).array().exp();
    
}


void Gaussian::gen_data(){
    std::vector<Matrix<double, Dynamic,Dynamic>> v;
    float low = Gaussian::ini_param("low");
    float high = Gaussian::ini_param("high");
    unsigned int n, p, t;
    n = Gaussian::ini_param("train");
    p = Gaussian::ini_param("pseudo");
    t = Gaussian::ini_param("test");
    /*
     *Generating Training points 
     */
    
    Gaussian::X = unigen(n,low,high);   
    save_data(Gaussian::X);

    Gaussian::noise_var =ini_param("noise_var");
    Gaussian::y = func_gen(Gaussian::X,Gaussian::noise_var);
    save_data(Gaussian::y);


    /*
     *Generating Pseudo points 
     */
    Gaussian::Xp = unigen(p,low,high);
    save_data(Gaussian::Xp );

    /*
     *Generating test points 
     */
    Gaussian::Xt = linspace(t,low,high);
    save_data(Gaussian::Xt);

}





std::pair<VectorXd,MatrixXd>  Gaussian::training(){ 
    /*
     *Computing and Kernel functions Knn, Kpp, Knp, Kpn
     */
    Matrix<double, Dynamic, Dynamic> Knn, Kpp, Knp, Kpn;
    Knn = kernel_fn(Gaussian::X,Gaussian::X);
    Kpp = kernel_fn(Gaussian::Xp,Gaussian::Xp);
    Knp = kernel_fn(Gaussian::X,Gaussian::Xp);
    Kpn = Knp.transpose();

    LLT<Matrix<double, Dynamic, Dynamic>> L_Kpp;
	L_Kpp.compute(Kpp);
    
    /*  
     * Calculating lambda + var 
     */
    Vector<double,Dynamic> lambda_diag(Gaussian::X.rows());
    for (int i=0;i<Gaussian::X.rows();i++){
        lambda_diag(i) = Knn(i,i) - Knp.row(i)*L_Kpp.solve(Knp.row(i).transpose());
    }
    Matrix<double, Dynamic, Dynamic> lambda = lambda_diag.asDiagonal();
    Matrix<double, Dynamic, Dynamic> lambda_var = lambda + Gaussian::noise_var * Matrix<double,Dynamic,Dynamic>::Identity(Gaussian::X.rows(),Gaussian::X.rows());
    
    /*
     * calculating Ktt and Ktp using test points xt
     */
    
    Matrix<double, Dynamic, Dynamic> Ktt, Ktp;
    Ktt = kernel_fn(Xt,Xt);
    Ktp = kernel_fn(Xt,Xp);
    
    /*  
     * Calculating Qm
     */

    LLT<Matrix<double, Dynamic, Dynamic>> L_lambdavar;
	L_lambdavar.compute(lambda_var);

    Matrix<double, Dynamic, Dynamic> Qp = Kpp + Knp.transpose() * L_lambdavar.solve(Knp);
    LLT<Matrix<double, Dynamic, Dynamic>> L_Qp;
    L_Qp.compute(Qp);


    /*
     *Computing mean and Covariance of predictive distribution
     */

    Vector<double, Dynamic> mu_pred = Ktp * L_Qp.solve(Kpn)*L_lambdavar.solve(Gaussian::y);
    Matrix<double, Dynamic, Dynamic> sigma_pred = Ktt - Ktp*(L_Kpp.solve(Ktp.transpose()) - L_Qp.solve(Ktp.transpose())) + noise_var * Matrix<double,Dynamic,Dynamic>::Identity(Gaussian::Xt.rows(),Gaussian::Xt.rows());
    
    // return {mu_pred, sigma_pred};
    return {mu_pred, sigma_pred}; 

}


