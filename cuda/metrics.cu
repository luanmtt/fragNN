#include "kernel.cuh"

#include <cstdio>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void confusion_init(Confusion& C, int n_classes){

    /*
        inicializa a matriz de confusão com zeros para todas as classes.

        exemplo: n_classes = 3 (gênero: masculino, feminino, unissex)
            TP = [0, 0, 0]
            FP = [0, 0, 0]
            FN = [0, 0, 0]
            TN = [0, 0, 0]
            total_correct = 0
            total_samples = 0
    */

    C.TP = new int[n_classes]();  // () zeros them
    C.FP = new int[n_classes]();
    C.FN = new int[n_classes]();
    C.TN = new int[n_classes]();
    C.total_correct = 0;
    C.total_samples = 0;
}


void confusion_free(Confusion& C){

    /*
        libera a memória alocada para a matriz de confusão.
        chame após terminar de usar as métricas para evitar memory leak.
    */

    delete[] C.TP;
    delete[] C.FP;
    delete[] C.FN;
    delete[] C.TN;
}


// ─────────────────────────────────────────────────────────────────────────────────────────────────
// accumulate


void confusion_accumulate(  Confusion& C,
                            float* result_probs,
                            int* result_labels,
                            int current_batch,
                            int n_classes){

    /*
        acumula os contadores TP, FP, FN para um batch de predições.
        faça isso para cada batch durante o evaluate().

        exemplo: batch de 4 amostras, 3 classes (gênero)
            result_probs = [0.1, 0.7, 0.2,   // amostra 0: predito classe 1
                            0.8, 0.1, 0.1,   // amostra 1: predito classe 0
                            0.3, 0.3, 0.4,   // amostra 2: predito classe 2
                            0.2, 0.6, 0.2]   // amostra 3: predito classe 1
            result_labels = [1, 0, 2, 0]      // classes reais

            resultado após acumular:
                classe 0: TP=1, FP=0, FN=1  (acertou amostra 1, errou amostra 3)
                classe 1: TP=1, FP=1, FN=0  (acertou amostra 0, errou amostra 3)
                classe 2: TP=1, FP=0, FN=0  (acertou amostra 2)
    */
    
    for(int i = 0; i < current_batch; i++){

        int predicted = 0;
        for(int c = 0; c < n_classes; c++){
            
            if(result_probs[i * n_classes + c] > result_probs[i * n_classes + predicted])
                predicted = c;

        }
        
        int actual = result_labels[i];
        for(int c = 0; c < n_classes; c++){
            
            if(actual == c && predicted == c) C.TP[c]++;
            if(actual != c && predicted == c) C.FP[c]++;
            if(actual == c && predicted != c) C.FP[c]++;

        }
        
        if(predicted == actual) C.total_correct++;
        C.total_samples++;
    }
}


void precision( Confusion& C,
                float* out,
                int n_classes){

    /*
        calcula a precision para cada classe: "das predições que fiz como classe X,
        quantas realmente eram classe X?"

        fórmula: precision[c] = TP[c] / (TP[c] + FP[c])

        exemplo: classe 0 (masculino)
            TP[0] = 50, FP[0] = 10
            precision[0] = 50 / (50 + 10) = 0.833

            → de todas as vezes que o modelo disse "masculino", 83.3% estavam corretas.
    */
    
    const float epsilon = 1e-18;
    for(int c = 0; c < n_classes; c++){
        out[c] = (float)C.TP[c] / (float)(C.TP[c] + C.FP[c] + epsilon);
    }

}


void recall(Confusion& C, 
            float* out,
            int n_classes){

    /*
        calcula o recall para cada classe: "das amostras que realmente são classe X,
        quantas o modelo conseguiu encontrar?"

        fórmula: recall[c] = TP[c] / (TP[c] + FN[c])

        exemplo: classe 0 (masculino)
            TP[0] = 50, FN[0] = 15
            recall[0] = 50 / (50 + 15) = 0.769

            → o modelo encontrou 76.9% de todos os masculinos que existiam.
    */

    const float epsilon = 1e-18;
    for(int c = 0; c < n_classes; c++){
        out[c] = (float)C.TP[c] / (float)(C.TP[c] + C.FN[c] + epsilon);
    }    

}


void f1(float* precision,
        float* recall,
        float* out,
        int n_classes){

    /*
        calcula o F1-score para cada classe: a média harmônica entre precision e recall.
        penaliza valores baixos em qualquer um dos dois.

        fórmula: F1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])

        exemplo: classe 0
            precision[0] = 0.833, recall[0] = 0.769
            F1[0] = 2 * 0.833 * 0.769 / (0.833 + 0.769) = 0.800

            → equilíbrio entre precision e recall. F1=1.0 é perfeito, F1=0.0 é péssimo.
    */
    
    const float epsilon = 1e-18;
    for(int c = 0; c < n_classes; c++){
        out[c] = 2.0f * precision[c] * recall[c] / (precision[c] + recall[c] + epsilon);
    }

}


float accuracy( Confusion& C){

    /*
        calcula a acurácia geral: "de todas as amostras, quantas o modelo acertou?"

        fórmula: accuracy = total_correct / total_samples

        exemplo:
            total_correct = 750, total_samples = 1000
            accuracy = 750 / 1000 = 0.75

            → o modelo acertou 75% de todas as predições.
            cuidado: acurácia pode ser enganosa em classes desbalanceadas.
            se 90% das amostras são classe 0, um modelo que sempre prediz classe 0
            terá 90% de acurácia mas será inútil.
    */
    
    const float epsilon = 1e-18;
    return (float)C.total_correct  / (float)(C.total_samples + epsilon);

}


float macro_avg(float* per_class, int n_classes){

    /*
        calcula a média macro: média aritmética simples entre todas as classes.
        todas as classes têm o mesmo peso, independente do número de amostras.

        exemplo: 3 classes
            per_class = [0.83, 0.79, 0.52]  (precision de cada classe)
            macro_avg = (0.83 + 0.79 + 0.52) / 3 = 0.713

            → se a classe 2 tem poucas amostras, ela influencia igual nas outras.
            (diferente de weighted avg, que daria mais peso à classe com mais amostras)
    */
    
    float sum = 0.0f;
    for(int c = 0; c < n_classes; c++){
        sum += per_class[c];
    }

    return sum / (float)n_classes;

}


// ─────────────────────────────────────────────────────────────────────────────────────────────────


struct ROCPoint {
    
    float fpr;
    float tpr;
};


void ROC(   float* probs,
            int* labels,
            int samples,
            int n_classes,
            const char* prefix){

    /*
        gera a curva ROC (Receiver Operating Characteristic) para cada classe
        e calcula a AUC (Area Under the Curve).

        a curva ROC plota TPR (taxa de verdadeiros positivos) vs FPR (taxa de falsos positivos)
        em diferentes limiares de confiança.

        para cada classe c:
            1. extrair as probabilidades da classe c e os rótulos verdadeiros
            2. ordenar por probabilidade decrescente
            3. variar o limiar de 1.0 até 0.0, calculando TPR e FPR em cada ponto
            4. salvar pontos em CSV para gráfico posterior
            5. calcular AUC (área sob a curva) pela regra dos trapézios

        exemplo: classe 0 (masculino), 5 amostras
            probs_c   = [0.9, 0.7, 0.6, 0.3, 0.1]
            is_pos    = [  1,   1,   0,   1,   0]  (amostra 0,1,3 são masculinos)

            varrendo o limiar:
                limiar=0.9: TPR=1/3=0.33, FPR=0/2=0.00
                limiar=0.7: TPR=2/3=0.67, FPR=0/2=0.00
                limiar=0.6: TPR=2/3=0.67, FPR=1/2=0.50
                limiar=0.3: TPR=3/3=1.00, FPR=1/2=0.50
                limiar=0.1: TPR=3/3=1.00, FPR=2/2=1.00

            AUC = 0.83 (bom — acima de 0.5 é melhor que aleatório)

        arquivos gerados: prefix_c0.csv, prefix_c1.csv, ...
        ex: "results/roc_gender_c0.csv"
    
    */  
    
    for(int c = 0; c < n_classes; c++){

        float* probs_c = new float[samples];
        int* is_pos = new int[samples];

        for(int i = 0; i < samples; i++){
            probs_c[i] = probs[i * n_classes + c];
            is_pos[i] = (labels[i] == c) ? 1 : 0;
        }
        
        for(int i = 0; i < samples; i++){
            for(int j = 0; j < samples; j++){

                if(probs[j] < probs[j+1]){
                    float tmp_p = probs[j];
                    probs[j] = probs[j+1];
                    probs[j+1] = tmp_p;

                    int tmp_l = is_pos[j];
                    is_pos[j] = is_pos[j+1];
                    is_pos[j+1] = tmp_l;

                }
            }
        }
 
        int P = 0;
        for(int i = 0; i < samples; i++) P += is_pos[i];
        int N = samples - P;
        
        ROCPoint* points = new ROCPoint[samples + 2];
        int n_points = 0;

        points[n_points++] = {0.0f, 0.0f};
        

        int tp = 0, fp = 0;
        for(int i = 0; i < samples; i++){

            if(is_pos[i]) 
                tp++;
            else          
                fp++;

            float tpr = (P > 0) ? (float)tp / (float)P : 0.0f;
            float fpr = (N > 0) ? (float)fp / (float)N : 0.0f;

            points[n_points++] = {fpr, tpr};
        }

        // ── 4. write CSV ──
        char filename[256];
        sprintf(filename, "%s_c%d.csv", prefix, c);

        FILE* f = fopen(filename, "w");
        fprintf(f, "fpr,tpr\n");

        for(int i = 0; i < n_points; i++)
            fprintf(f, "%.6f,%.6f\n", points[i].fpr, points[i].tpr);

        fclose(f);

        // ── 5. AUC (trapezoidal rule) ──
        float auc = 0.0f;
        for(int i = 1; i < n_points; i++){
            float dx = points[i].fpr - points[i - 1].fpr;
            float dy = points[i].tpr + points[i - 1].tpr;
            auc += dx * dy / 2.0f;
        }

        printf("  class %d: AUC = %.4f\n", c, auc);

        delete[] probs_c;
        delete[] is_pos;
        delete[] points;

    }
}


// ─────────────────────────────────────────────────────────────────────────────────────────────────


void metrics_router(metrics_used flags,
                    Confusion& C,
                    float* probs,
                    int* labels,
                    int samples,
                    int n_classes,
                    const char* prefix){

    /*
        roteador principal de métricas. verifica quais métricas foram solicitadas
        no struct metrics_used e chama as funções correspondentes.

        exemplos de uso:

            // só acurácia
            metrics_used flags = {false, false, false, false, true};
            metrics_router(flags, C, probs, labels, 1000, 3, "results/roc_gender");

            // tudo
            metrics_used flags = {true, true, true, true, true};
            metrics_router(flags, C, probs, labels, 1000, 3, "results/roc_gender");

            // precision + ROC
            metrics_used flags = {true, false, false, true, false};
            metrics_router(flags, C, probs, labels, 1000, 3, "results/roc_gender");

        saída no terminal:
            ────────────────────────────────────────
              accuracy:    0.7500
              precision:   0.7200 (macro)
              recall:      0.6800 (macro)
              f1:          0.6990 (macro)

              per-class:
              class           prec      rec       f1
              0             0.8333   0.7692   0.8000
              1             0.7900   0.8800   0.8325
              2             0.5200   0.4500   0.4824

              ROC curves:
                class 0: AUC = 0.8320
                class 1: AUC = 0.9100
                class 2: AUC = 0.6200
    */
                    
    float* recall_ = new float[n_classes];
    float* precision_ = new float[n_classes];
    float* f1_ = new float[n_classes];


        
    printf("──────────────────────────────────────────────────────────────────────");
    
    
    if(flags.ACCURACY){
        float acc = accuracy(C);
        printf("  accuracy:    %4f\n", acc);
    }


    if(flags.PRECISION){

        precision(C, precision_, n_classes);
        float macro_precision = macro_avg(precision_, n_classes);
        
        printf("  precision:   %4f (macro)\n", macro_precision);
     }
    
    if(flags.RECALL){
            
        recall(C, recall_, n_classes);
        float macro_recall = macro_avg(recall_, n_classes);
        printf("  accuracy:    %4f (macro)\n", macro_recall);
    }
    
    if(flags.F1){
        
        // obrigatórios

        if(!flags.RECALL){
            recall(C, recall_, n_classes);
        } 

        if(!flags.PRECISION){
            precision(C, precision_, n_classes);
        } 
        
        f1(precision_, recall_, f1_, n_classes);
        float macro_f1 = macro_avg(f1_, n_classes);
        printf("  f1:       %.4f (macro)\n", macro_f1);
    }


    if(flags.PRECISION || flags.PRECISION || flags.F1){

        printf("\n  per-class:\n");
        printf("  %-12s %8s %8s %8s\n", "class", "prec", "rec", "f1");
        for(int c = 0; c < n_classes; c++){
            printf("  %-12d %8.4f %8.4f %8.4f\n",
                c,
                flags.PRECISION ? precision_[c] : 0.0f,
                flags.RECALL    ? recall_[c]    : 0.0f,
                flags.F1       ? f1_[c]        : 0.0f);
        }
    }

    if(flags.ROC){
        printf("\n  ROC curves:\n");
        ROC(probs, labels, samples, n_classes, prefix);
    }

    delete[] precision_;
    delete[] recall_;
    delete[] f1_;
    

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
