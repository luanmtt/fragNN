#include <iostream>

#include "src/headers/dataloader.hpp"
//#include "src/headers/nn.hpp"



void div(){
    cout << "───────────────────────────────────────────────────────────────────────\n";
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


int main(){
    
    cout << "Iniciando FragNN: \n";

    div();
    cout << "Testando load_csv: \n";
            
    Dataset train_t = load_csv(TRAIN_PATH);
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    

    div();
    cout << "Teste dos 5 primeiros elementos: \n"; 

    for(int i = 0; i < 5; i++){
        cout << "[" << i << "]: " << train_t.X[train_t.num_rows - i] << "\n";
    }


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    

    div();
    cout << "Liberando memória. Fim do programa. \n";

    free_data(train_t);

    return 0;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
