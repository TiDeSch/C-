#include <iostream>
#include <vector>
#include <fstream>
#include <cmath> 
#include <algorithm>
#include <tuple>
//We are both creating a matrix with absolute difference and vectors for S, I, R using dt and dt/2. This is done so we can get loop over the 10 different dt, but also get a vector of the difference between dt and dt/2 with one dt.
//Be aware, if you want to create results for task 1 and 2, and want S, I and R half and difference. Then the vector of dt_values must contain delta_t 



// Function to take a step in the SIR model
// state: vector of S, I, R
// beta: infection rate
// gamma: recovery rate
// dt: time step

//todo: implement the SIR model
std::vector<float> take_step(std::vector<float> state, float beta, float gamma, float dt, float N) {
    float S = state[0];
    float I = state[1];
    float R = state[2];
    
    float dS = -beta * S * I / N * dt;
    float dI = (beta * S * I / N - gamma * I) * dt;
    float dR = gamma * I * dt;

    S += dS;
    I += dI;
    R += dR;

    return {S, I, R};
}
//getting the coloumn from the matrix containg days, S, I and R
std::vector<float> get_column(const std::vector<std::vector<float>>& matrix, int col_index) {
        std::vector<float> column;
        for (const auto& row : matrix) {  
            if (col_index < row.size()) { 
                column.push_back(row[col_index]);  
            }
        }
        return column;
}
//Subtract to vector from each other e.g. S^dt and S^dt/2
void subtract(const std::vector<float>& new_state, 
              const std::vector<float>& new_state_half, 
              std::vector<float>& res) 
{
    for (size_t i = 0; i < new_state.size(); i++) {
        res[i] = std::fabs(new_state[i] - new_state_half[i]);
    }
}
//Finding the maximum value of the matrix containing the subtracted value from each other
void compute_max(const std::vector<float>& differences, const std::string& label) {
    auto max_iter = std::max_element(differences.begin(), differences.end());
    float max_value = *max_iter;
    int max_index = std::distance(differences.begin(), max_iter);
    
    std::cout << "Maximum " << label << " difference: " << max_value 
              << " at index " << max_index << "\n";
}


//======================================================================================================
//======================== Main function ===============================================================
//======================================================================================================

int main() {
    // define the parameters
    float beta = 0.2;
    float gamma = 0.1;
    float delta_t = 1;
    int days_total = 200;
    float N = 1000;

    //task 3
    std::vector<float> dt_values = {1.0,0.2154,0.0464,0.01,0.00215,0.000464,0.0001,0.0000215,0.00000464, 0.000001};
    std::vector<float> max_differences(10, 0.0f); 

    std::ofstream file3("max_overall_differences.csv");
    file3 << "dt,Max_Value, s, I, R\n";

    for (size_t dt_index = 0; dt_index < dt_values.size(); ++dt_index) {
        float dt = dt_values[dt_index];
        
        // TODO: Define the parameters of the SIR model
        float S = 0.999 * N;
        float I = 0.001 * N; 
        float R = N - S - I;
        float S_half = S;
        float I_half = I; 
        float R_half = R;

        std::vector<std::vector<float>> matrix_res;
        std::vector<float> new_state(3, 0.0f);
        std::vector<float> new_state_half(3, 0.0f);
        std::vector<float> res(3, 0.0f);

        std::ofstream file, file1, file2;
        if (dt == delta_t) {
            // TODO: Save the results to a file
            file.open("SIR_results.csv");
            file1.open("SIR_half_results.csv");
            file2.open("SIR_abs_diff.csv");

            file << "Day,Susceptible,Infected,Recovered\n";
            file1 << "Day,Susceptible,Infected,Recovered\n";
            file2 << "Day,Abs_Diff_S,Abs_Diff_I,Abs_Diff_R\n";
        }

        for (int day = 0; day <= days_total; ++day) {
            if (dt == delta_t) {
                file << day << "," << S << "," << I << "," << R << "\n";
                file1 << day << "," << S_half << "," << I_half << "," << R_half << "\n";
            }
            
            // TODO: implement the SIR model
            float t = 0;
            float t_half = 0;
            while (t < 1.0f) {
                new_state = take_step({S, I, R}, beta, gamma, dt, N);
                S = new_state[0]; 
                I = new_state[1];
                R = new_state[2];
                t += dt;
                //Finding the half values
                for (int i = 0; i < 2; ++i) {
                    if (t_half < 1.0f) {
                        new_state_half = take_step({S_half, I_half, R_half}, beta, gamma, dt / 2, N);
                        S_half = new_state_half[0]; 
                        I_half = new_state_half[1]; 
                        R_half = new_state_half[2];
                        t_half += dt / 2;
                    }
                }
            }

            subtract(new_state, new_state_half, res);
            matrix_res.push_back(res);
          

            if (dt == delta_t) {
                file2 << day << "," << res[0] << "," << res[1] << "," << res[2] << "\n";
            }
        }

        // TODO: Save the results to a file
        if (dt == delta_t) {
            file.close();
            file1.close();
            file2.close();
        }

        float max_diff = 0.0f;
        for (int i = 0; i < 3; ++i) {
            std::vector<float> differences = get_column(matrix_res, i);
            float max_value = *std::max_element(differences.begin(), differences.end());
            max_diff = std::max(max_diff, max_value);
        }
        
        max_differences[dt_index] = max_diff;
        // Find the max difference for this dt
        std::vector<float> max_S = get_column(matrix_res, 0);
        std::vector<float> max_I = get_column(matrix_res, 1);
        std::vector<float> max_R = get_column(matrix_res, 2);
        
        float max_diff_S = *std::max_element(max_S.begin(), max_S.end());
        float max_diff_I = *std::max_element(max_I.begin(), max_I.end());
        float max_diff_R = *std::max_element(max_R.begin(), max_R.end());
        
        file3 << dt << "," << max_diff << "," << max_diff_S << "," << max_diff_I << "," << max_diff_R << "\n";
    }


    file3.close();

    return 0;
}

