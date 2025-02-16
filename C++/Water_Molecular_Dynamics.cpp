#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cassert>
#include <math.h>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <omp.h>

const double deg2rad = acos(-1)/180.0;    // pi/180 for changing degs to radians
double accumulated_forces_bond  = 0.;     // Checksum: accumulated size of forces
double accumulated_forces_angle = 0.;     // Checksum: accumulated size of forces
double accumulated_forces_non_bond = 0.;  // Checksum: accumulated size of forces
constexpr size_t nClosest = 8;            // Number of closest neighbors to consider in neighbor list. 

class Vec3 {
public:
    double x, y, z;
    Vec3(double x, double y, double z): x(x), y(y), z(z) {}
    double mag2() const{
        return x*x+y*y+z*z;
    }
    double mag() const{
        return sqrt(mag2());
    }
    Vec3 operator-(const Vec3& other) const{
        return {x - other.x, y - other.y, z - other.z};
    }
    Vec3 operator+(const Vec3& other) const{
        return {x + other.x, y + other.y, z + other.z};
    }
    Vec3 operator*(double scalar) const{
        return {scalar*x, scalar*y, scalar*z};
    }
    Vec3 operator/(double scalar) const{
        return {x/scalar, y/scalar, z/scalar};
    }
    Vec3& operator+=(const Vec3& other){
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
    Vec3& operator-=(const Vec3& other){
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }
    Vec3& operator*=(double scalar){
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }
    Vec3& operator/=(double scalar){
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }
};
Vec3 operator*(double scalar, const Vec3& y){
    return y*scalar;
}
Vec3 cross(const Vec3& a, const Vec3& b){
    return { a.y*b.z-a.z*b.y,
             a.z*b.x-a.x*b.z,
             a.x*b.y-a.y*b.x };
}
double dot(const Vec3& a, const Vec3& b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

class Atoms {
public:
    double mass;
    double ep;
    double sigma;
    double charge;
    std::string name;
    std::vector<Vec3> p,v,f;

    Atoms(double mass, double ep, double sigma, double charge, std::string name, size_t N_identical) 
    : mass{mass}, ep{ep}, sigma{sigma}, charge{charge}, name{name}, 
      p{N_identical, {0,0,0}}, v{N_identical, {0,0,0}}, f{N_identical, {0,0,0}}
    {}
};

class Bond {
public:
    double K;
    double L0;
    size_t a1, a2;
};

class Angle {
public:
    double K;
    double Phi0;
    size_t a1, a2, a3;
};

class Molecules {
public:
    std::vector<Atoms> atoms;
    std::vector<Bond> bonds;
    std::vector<Angle> angles;
    std::vector<std::vector<size_t>> neighbours;
    size_t no_mol;

    Molecules(std::vector<Atoms> atoms, std::vector<Bond> bonds, std::vector<Angle> angles, size_t no_mol)
    : atoms{atoms}, bonds{bonds}, angles{angles}, neighbours{no_mol}, no_mol{no_mol}
    {}
};

class System {
public:
    Molecules molecules;
    double time = 0;

    System(std::vector<Atoms> atoms, std::vector<Bond> bonds, std::vector<Angle> angles, size_t no_mol)
    : molecules{atoms, bonds, angles, no_mol} {}
};

class Sim_Configuration {
public:
    size_t steps = 100000;
    size_t no_mol = 10000;
    double dt = 0.0005;
    size_t data_period = 100;
    std::string filename = "trajectory.txt";

    Sim_Configuration(std::vector <std::string> argument){
        for (size_t i = 1; i<argument.size() ; i += 2){
            std::string arg = argument.at(i);
            if(arg=="-h"){
                std::cout << "MD -steps <number of steps> -no_mol <number of molecules>"
                          << " -fwrite <io frequency> -dt <size of timestep> -ofile <filename> \n";
                exit(0);
                break;
            } else if(arg=="-steps"){
                steps = std::stoi(argument[i+1]);
            } else if(arg=="-no_mol"){
                no_mol = std::stoi(argument[i+1]);
            } else if(arg=="-fwrite"){
                data_period = std::stoi(argument[i+1]);
            } else if(arg=="-dt"){
                dt = std::stof(argument[i+1]);
            } else if(arg=="-ofile"){
                filename = argument[i+1];
            } else{
                std::cout << "---> error: the argument type is not recognized \n";
            }
        }

        dt /= 1.57350;
    }
};



void BuildNeighborList(System& sys){
    std::vector<double> distances2(sys.molecules.no_mol);
    std::vector<size_t> index(sys.molecules.no_mol);
    
    
    for(size_t i = 0; i < sys.molecules.no_mol; i++){
        sys.molecules.neighbours[i].clear();
        
        #pragma omp simd
        for (size_t j = 0; j < sys.molecules.no_mol; j++) {
            Vec3 dp = sys.molecules.atoms[0].p[i] - sys.molecules.atoms[0].p[j];
            distances2[j] = dp.mag2();
            index[j] = j;
        }
        distances2[i] = 1e99;

        size_t target_num = std::min(nClosest, sys.molecules.no_mol-1);

        auto lambda_compare = [&](size_t &a, size_t &b) { return distances2[a] < distances2[b]; };
        
        std::partial_sort(index.begin(), index.begin() + target_num, index.end(), lambda_compare);

        for (size_t j=0; j < target_num; j++) {
            auto& k = index[j];
            if (k < i) {
                auto& neighbours = sys.molecules.neighbours[k];
                if (std::find(neighbours.begin(), neighbours.end(), i) == neighbours.end())
                    sys.molecules.neighbours[i].push_back(k);
            } else {
                sys.molecules.neighbours[i].push_back(k);
            }
        }
    }
}


void UpdateBondForces(System& sys){
    Molecules& molecule = sys.molecules;
    for (Bond& bond : molecule.bonds){
        auto& atom1 = molecule.atoms[bond.a1];
        auto& atom2 = molecule.atoms[bond.a2];
        
        //#pragma omp simd
        for (size_t i = 0; i < molecule.no_mol; i++) {
            Vec3 dp  = atom1.p[i] - atom2.p[i];
            Vec3 f   = -bond.K*(1-bond.L0/dp.mag())*dp;
            atom1.f[i] += f;
            atom2.f[i] -= f;
            accumulated_forces_bond += f.mag();
        }
    }
}

void UpdateAngleForces(System& sys){
    Molecules& molecule = sys.molecules;
    for (Angle& angle : molecule.angles){
        auto& atom1 = molecule.atoms[angle.a1];
        auto& atom2 = molecule.atoms[angle.a2];
        auto& atom3 = molecule.atoms[angle.a3];
        
        //#pragma omp simd
        for (size_t i = 0; i < molecule.no_mol; i++) {
            Vec3 d21 = atom2.p[i] - atom1.p[i];
            Vec3 d23 = atom2.p[i] - atom3.p[i];

            double norm_d21 = d21.mag();
            double norm_d23 = d23.mag();
            double phi = acos(dot(d21, d23) / (norm_d21 * norm_d23));

            Vec3 c21_23 = cross(d21, d23);
            Vec3 Ta = cross(d21, c21_23) / cross(d21, c21_23).mag();
            Vec3 Tc = cross(c21_23, d23) / cross(c21_23, d23).mag();

            Vec3 f1 = Ta * (angle.K * (phi - angle.Phi0) / norm_d21);
            Vec3 f3 = Tc * (angle.K * (phi - angle.Phi0) / norm_d23);

            
            atom1.f[i] += f1;
            
            atom2.f[i] -= (f1 + f3);
            
            atom3.f[i] += f3;

            accumulated_forces_angle += f1.mag() + f3.mag();
        }
    }
}

void UpdateNonBondedForces(System& sys){
    Molecules& molecule = sys.molecules;
    
    //#pragma omp parallel for collapse(4)
    for (size_t i = 0; i < molecule.no_mol; i++)
    for (auto& j : molecule.neighbours[i])
    for (size_t k = 0; k < molecule.atoms.size(); k++)
    for (size_t l = 0; l < molecule.atoms.size(); l++) {
        auto& atom1 = molecule.atoms[k];
        auto& atom2 = molecule.atoms[l];
        double ep = sqrt(atom1.ep * atom2.ep);
        double sigma2 = pow(0.5 * (atom1.sigma + atom2.sigma), 2);
        double KC = 80 * 0.7;
        double q = KC * atom1.charge * atom2.charge;

        Vec3 dp = atom1.p[i] - atom2.p[j];
        double r2 = dp.mag2();
        double r = sqrt(r2);

        double sir = sigma2 / r2;
        double sir3 = sir * sir * sir;
        Vec3 f = (ep * (12 * sir3 * sir3 - 6 * sir3) * sir + q / (r * r2)) * dp;

        
        atom1.f[i] += f;
        
        
        atom2.f[j] -= f;
        
        
        accumulated_forces_non_bond += f.mag();
    }
}



void UpdateKDK(System &sys, Sim_Configuration &sc){
    Molecules& molecule = sys.molecules;
    for (size_t k = 0; k < molecule.atoms.size(); k++) {
        auto& atom = molecule.atoms[k];
        for (size_t i = 0; i < molecule.no_mol; i++) {
            atom.v[i] += sc.dt/atom.mass*atom.f[i];
            atom.f[i]  = {0,0,0};
            atom.p[i] += sc.dt* atom.v[i];
        }
    }

    sys.time += sc.dt;
}

System MakeWater(size_t N_molecules){
    const double L0 = 0.09584;
    const double angle = 104.45*deg2rad;

    Atoms Oatom(16, 0.65, 0.31, -0.82, "O", N_molecules);
    Atoms Hatom1(1, 0.18828, 0.238, 0.41, "H", N_molecules);
    Atoms Hatom2(1, 0.18828, 0.238, 0.41, "H", N_molecules);

    std::vector<Bond> waterbonds = {
        { .K = 20000, .L0 = L0, .a1 = 0, .a2 = 1},
        { .K = 20000, .L0 = L0, .a1 = 0, .a2 = 2}
    };

    std::vector<Angle> waterangle = {
        { .K = 1000, .Phi0 = angle, .a1 = 1, .a2 = 0, .a3 = 2 }
    };

    double phi = acos(-1) * (sqrt(5.) - 1.);
    double radius = sqrt(N_molecules)*0.15;
    for (size_t i = 0; i < N_molecules; i++){
        double y = 1 - (i / (N_molecules - 1.));
        double r = sqrt(1 - y * y);
        double theta = phi * i;

        double x = cos(theta) * r;
        double z = sin(theta) * r;

        Vec3 P0{x*radius, y*radius, z*radius};
        Oatom.p[i]  = {P0.x, P0.y, P0.z};
        Hatom1.p[i] = {P0.x+L0*sin(angle/2), P0.y+L0*cos(angle/2), P0.z};
        Hatom2.p[i] = {P0.x-L0*sin(angle/2), P0.y+L0*cos(angle/2), P0.z};
    }

    std::vector<Atoms> atoms {Oatom, Hatom1, Hatom2};
    return System(atoms, waterbonds, waterangle, N_molecules);
}

void WriteOutput(System& sys, std::ofstream& file){
    Molecules& molecule = sys.molecules;
    for (size_t k = 0; k < molecule.atoms.size(); k++) {
        auto& atom = molecule.atoms[k];
        for (size_t i = 0; i < molecule.no_mol; i++) {
            file << sys.time << " " << atom.name << " "
                << atom.p[i].x << " "
                << atom.p[i].y << " "
                << atom.p[i].z << '\n';
        }
    }
}

int main(int argc, char* argv[]){
    Sim_Configuration sc({argv, argv+argc});

    System sys  = MakeWater(sc.no_mol);
    std::ofstream file(sc.filename);

    WriteOutput(sys, file);

    auto tstart = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < sc.steps; step++){
        if (step % 10 == 0)
            BuildNeighborList(sys);

        UpdateBondForces(sys);
        UpdateAngleForces(sys);
        UpdateNonBondedForces(sys);
        UpdateKDK(sys, sc);

        if (step % sc.data_period == 0)
        {
            WriteOutput(sys, file);
        }
    }

    auto tend = std::chrono::high_resolution_clock::now();

    std::cout <<  "Accumulated forces Bonds   : "  << std::setw(9) << std::setprecision(5)
              << accumulated_forces_bond << "\n";
    std::cout <<  "Accumulated forces Angles  : "  << std::setw(9) << std::setprecision(5)
              << accumulated_forces_angle << "\n";
    std::cout <<  "Accumulated forces Non-bond: "  << std::setw(9) << std::setprecision(5)
              << accumulated_forces_non_bond << "\n";
    std::cout << "Elapsed total time:       " << std::fixed << std::setw(9) << std::setprecision(4)
              << (tend - tstart).count()*1e-9 << "\n";
}