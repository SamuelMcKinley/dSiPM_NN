//UI for dSiPM_NN

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

// Check if SPAD Size is legitimate
bool SPAD_Check(std::string s_check){
    for (int j=1; j<160001; j++){
        std::string s_real= std::to_string(j) + "x" + std::to_string(j);
        if (s_real == s_check){
            return true;
        }

    }
    return false;
}


int main(){
    double E_min, E_max, step;
    int total_events, group_size;
    std::string particle;
    int n_spads;

    std::cout << "Enter Lowest Energy (GeV): ";
    std::cin >> E_min;

    std::cout << "Enter Highest Energy (GeV): ";
    std::cin >> E_max;

    std::cout << "Enter Step Size (GeV): ";
    std::cin >> step;

    // Check if Step Size is valid
    double E_difference = E_max - E_min;
    if (std::fmod(E_difference,step) != 0){
        std::cout << "Step Size not compatible with energy range. Exiting...\n";
        return 1;
    }

    std::cout << "Enter Total Number of Events: ";
    std::cin >> total_events;

    std::cout << "Enter Group Size (must divide " << total_events << " with no remainder): ";
    std::cin >> group_size;

    // Check if Group Size is valid
    if (std::fmod(total_events, group_size) != 0){
        std::cout << "Total Events not divisible by Group Size. Exiting...\n";
        return 1;
    }

    std::cout << "Enter Particle Type (pi+, e+, etc.): ";
    std::cin >> particle;

    // Checking if Particle is valid
    std::string allowed_particles[] = {
        "pi+", "pi-", "pi0", "e+", "e-", "mu+", "mu-", "proton", "gamma"
    };

    int n_particles = sizeof(allowed_particles) / sizeof(allowed_particles[0]);

    bool isParticle = false;
    for (int i=0; i < n_particles; i++){
        if (particle == allowed_particles[i]){
            isParticle = true;
            break;
        }
    }

    if (!isParticle){
        std::cout << "Particle " << particle << " not valid. Allowed particles are:\n";
        for (int i=0; i < n_particles; i++){
            std::cout << "'"<< allowed_particles[i] << "'  ";
        }
        std::cout << "\n";
        return 1;
    }

    std::cout << "How many SPAD Sizes?: ";
    std::cin >> n_spads;

    // For every SPAD Size, prompt to enter and check if legitimate
    std::vector<std::string> spad_sizes(n_spads);
    for (int i = 0; i < n_spads; i ++){
        std::cout << "Enter SPAD Size (e.g. '50x50') " << i+1 << ": ";
        std::string test;
        std::cin >> test;
        if (SPAD_Check(test) == true){
            spad_sizes[i]=test;
        }else {
            i = i - 1;
            std::cout << "Not valid SPAD Size format\n";
        }
        
 
    }

    // Print summary
    std::cout << "\n--- Simulation Parameters ---\n";
    std::cout << "Energy: " << E_min << " → " << E_max 
              << " step " << step << " GeV\n";
    std::cout << "Total events: " << total_events 
              << " (group size " << group_size << ")\n";
    std::cout << "Particle: " << particle << "\n";
    std::cout << "SPAD sizes: ";
    for (auto &s : spad_sizes) std::cout << s << " ";
    std::cout << "\n";


    std::cout << "Do you want to proceed? Y/N?";

    char proceed_check;
    std::cin >> proceed_check;
        if (proceed_check == 'Y' || proceed_check == 'y'){
            std::cout << "Proceeding\n";
        } else if (proceed_check == 'N' || proceed_check == 'n'){
            std::cout << "Breaking\n";
            return 1;
        } else {
            std::cout << "Answer not recognized. Provide Y or N\n";
        }


    // Check if user wants to run clear_files.sh
    std::cout << "\nDo you want to clear files of previous jobs? Y/N\n?";
    char answer;
    std::cin >> answer;
    while (true){
        if (answer == 'Y' || answer == 'y'){
            std::system("./clear_files.sh");
            std::cout << "Successfully cleared files\n";
            break;
        } else if (answer == 'N' || answer == 'n'){
            break;
        } else{
            std::cout << "Answer not recognized. Provide Y or N\n";
            std::cin >> answer;
        }
    }

    std::ostringstream cmd;
    cmd << "./masterTrain.sh "
        << E_min << " "
        << E_max << " "
        << step << " "
        << total_events << " "
        << group_size << " "
        << particle;

    for (const auto& spad : spad_sizes){
        cmd << " " << spad;
    }

    int ret = std::system(cmd.str().c_str());
    if (ret != 0){
        std::cerr << "Error: masterTrain.sh failed with code " << ret << "\n";
    }


    std::cout << "\n\nmasterTrain.sh edited and started\n";
    return 0;


}
