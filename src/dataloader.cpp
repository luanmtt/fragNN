#include "headers/dataloader.hpp"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


Dataset load_csv(const string &path){
    
    Dataset data_t = {  nullptr,
                        nullptr,
                        nullptr,
                        0,
                        0
                     };

    ifstream file(path);    
        
    if(!file.is_open()){
        cout << "Erro na leitura do arquivo: " << path << "\n";
        return data_t;
    }
        
    Dataset* dataset_t;

    vector<float>X_temp = {};
    vector<int>y_gender_temp = {};
    vector<int>y_accords_temp = {};
    
    int num_rows = 0;
    int num_features = 0;
    
    string line = {};
    int col = -1; // começa em 0


    while(getline(file, line)){
        //@{

        if(line.empty()) continue; 

        stringstream ss(line);
        string token;
        vector<float>row = {};
                
        while(getline(ss, token, ',')){
            row.push_back(stof(token));
        }
       
        if(row.empty()) continue;

        if(col == -1){
            
            col = (int)row.size();
            num_features = col - 2;
        }       

        int gender = (int)row[col - 2];
        int accord = (int)row[col - 1];
        
        for(int i = 0; i < num_features; i++){

            X_temp.push_back(row[i]);
        }

        y_gender_temp.push_back(gender);
        y_accords_temp.push_back(accord);
        
        num_rows++;

        //@}
    }

    file.close();


    data_t.X = new float[num_rows * num_features];
    data_t.y_gender = new int[num_rows];
    data_t.y_accord = new int[num_rows];
    data_t.num_rows = num_rows;
    data_t.num_features = num_features;


    for(int i = 0; i < num_rows * num_features; i++) 
        data_t.X[i] = X_temp[i];

    for(int i = 0; i < num_rows; i++) 
        data_t.y_gender[i] = y_gender_temp[i];
    
    for(int i = 0; i < num_rows; i++) 
        data_t.y_accord[i] = y_accords_temp[i];


    cout << "Foram carregadas " << num_rows << " linhas, e coletadas " << num_features << " features.\n";


    return data_t;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void free_data(Dataset& data_t){

    delete[] data_t.X;
    delete[] data_t.y_gender;
    delete[] data_t.y_accord;

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
