#include "embellish.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


static const char* SEP_CHAR = "─";
static const char* BAR_CHAR = "━";


static std::string repeat(const char* s, int n){

    std::string result;
    for(int i = 0; i < n; i++)
        result += s;
    return result;
}


// ─────────────────────────────────────────────────────────────────────────────────────────────────


std::string sep(int n){

    /*
        retorna uma string com n caracteres '─' seguidos de newline.

        exemplo: sep(20)
            → "────────────────────\n"
    */

    return repeat(SEP_CHAR, n) + "\n";
}


std::string sepComment(int n, const char* comment){

    /*
        retorna um separador com um comentário centralizado.

        exemplo: sepComment(40, "Epoch 1/50")
            → "────────────── Epoch 1/50 ──────────────\n"
    */

    int left  = n / 2;
    int right = n - left;

    return repeat(SEP_CHAR, left) + " " + comment + " " + repeat(SEP_CHAR, right) + "\n";
}


std::string sepBar(int n){

    /*
        retorna uma string com n caracteres '━' seguidos de newline.
        usado para separadores de destaque (ex: títulos de epoch).

        exemplo: sepBar(20)
            → "━━━━━━━━━━━━━━━━━━━━\n"
    */

    return repeat(BAR_CHAR, n) + "\n";
}


std::string sepBarComment(int n, const char* comment){

    /*
        retorna um separador de destaque com comentário centralizado.

        exemplo: sepBarComment(40, "Epoch 1/50")
            → "━━━━━━━━━━━━━━ Epoch 1/50 ━━━━━━━━━━━━━━\n"
    */

    int left  = n / 2;
    int right = n - left;

    return repeat(BAR_CHAR, left) + " " + comment + " " + repeat(BAR_CHAR, right) + "\n";
}


// ─────────────────────────────────────────────────────────────────────────────────────────────────
// cores ANSI


void hex_to_rgb(const char* hex, int& r, int& g, int& b){

    /*
        converte uma cor hex para os valores RGB.

        exemplo: hex_to_rgb("#F1BEB0", r, g, b)
            → r=241, g=190, b=176
    */

    const char* h = hex[0] == '#' ? hex + 1 : hex;
    char buf[3] = {0};

    buf[0] = h[0]; buf[1] = h[1]; r = (int)strtol(buf, nullptr, 16);
    buf[0] = h[2]; buf[1] = h[3]; g = (int)strtol(buf, nullptr, 16);
    buf[0] = h[4]; buf[1] = h[5]; b = (int)strtol(buf, nullptr, 16);
}


std::string colorPrint(const char* text, const char* hex){

    /*
        retorna o texto envolto em códigos de cor ANSI (256 cores).

        exemplo: colorPrint("olá", "#F1BEB0")
            → "\033[38;2;241;190;176molá\033[0m"
    */

    int r, g, b;
    hex_to_rgb(hex, r, g, b);

    char buf[512];
    snprintf(buf, sizeof(buf), "\033[38;2;%d;%d;%dm%s\033[0m", r, g, b, text);
    return std::string(buf);
}


std::string stylePrint(const char* text, const char* hex, bool bold, bool italic, bool underline){

    /*
        retorna o texto com estilo (negrito, itálico, sublinhado) e/ou cor.

        exemplo: stylePrint("atenção", "#78a9c7", true, false, true)
            → "\033[1;4;38;2;120;169;199matenção\033[0m"
    */

    char codes[128];
    int pos = 0;

    codes[pos++] = '\033';
    codes[pos++] = '[';

    bool first = true;

    if(bold){
        if(!first) codes[pos++] = ';';
        codes[pos++] = '1';
        first = false;
    }
    if(italic){
        if(!first) codes[pos++] = ';';
        codes[pos++] = '3';
        first = false;
    }
    if(underline){
        if(!first) codes[pos++] = ';';
        codes[pos++] = '4';
        first = false;
    }

    if(hex){
        int r, g, b;
        hex_to_rgb(hex, r, g, b);
        pos += snprintf(codes + pos, sizeof(codes) - pos, ";38;2;%d;%d;%d", r, g, b);
    }

    codes[pos++] = 'm';

    int text_len = snprintf(codes + pos, sizeof(codes) - pos, "%s", text);
    pos += text_len;

    codes[pos++] = '\033';
    codes[pos++] = '[';
    codes[pos++] = '0';
    codes[pos++] = 'm';
    codes[pos]   = '\0';

    return std::string(codes);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
