#pragma once

#include <string>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// embellish — utilitários de estética para terminal


std::string sep(int n);

std::string sepComment(int n, const char* comment);

std::string sepBar(int n);

std::string sepBarComment(int n, const char* comment);

void hex_to_rgb(const char* hex, int& r, int& g, int& b);

std::string colorPrint(const char* text, const char* hex);

std::string stylePrint(const char* text,
                       const char* hex = nullptr,
                       bool bold = false,
                       bool italic = false,
                       bool underline = false);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
