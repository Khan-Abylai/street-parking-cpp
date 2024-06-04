#pragma once

#include <vector>
#include <unordered_map>
#include <string>

enum class CountryCode {
    KZ,
    KG,
    UZ,
    RU,
    BY,
    GE,
    AM,
    AZ
};

class TemplateMatching {

private:

    const std::vector<std::vector<std::string>> SQUARE_TEMPLATES_HALF_KZ{{"999", "99AAA"},
                                                                         {"999", "99AA"},
                                                                         {"99",  "99AA"},
                                                                         {"A99", "9999"}};

    const std::unordered_map<CountryCode, std::string> COUNTRY_TO_STRING{
            {CountryCode::KZ, "KZ"},
            {CountryCode::KG, "KG"},
            {CountryCode::UZ, "UZ"},
            {CountryCode::RU, "RU"},
            {CountryCode::BY, "BY"},
            {CountryCode::GE, "GE"},
            {CountryCode::AM, "AM"},
            {CountryCode::AZ, "AZ"},
    };

    const std::unordered_map<std::string, CountryCode> COUNTRY_TEMPLATES{
            {"AA99AAA",   CountryCode::KZ},
            {"999AAA99",  CountryCode::KZ},
            {"999AA99",   CountryCode::KZ},
            {"99AA99",    CountryCode::KZ},
            {"999AAA",    CountryCode::KZ},
            {"A999AAA",   CountryCode::KZ},
            {"A999AA",    CountryCode::KZ},
            {"A999AA99",  CountryCode::RU},
            {"AA99999",   CountryCode::RU},
            {"A999AA999", CountryCode::RU},
            {"AA999A99",  CountryCode::RU},
            {"999A99999", CountryCode::RU},
            {"99999AAA",  CountryCode::KG},
            {"99999AA",   CountryCode::KG},
            {"99A999AA",  CountryCode::UZ},
            {"A999999",   CountryCode::KZ},
            {"AAA9999",   CountryCode::KZ},
            {"AA999",     CountryCode::KZ},
            {"999999",    CountryCode::KZ},
            {"999999A",   CountryCode::KG},
            {"AAAA9999",  CountryCode::KG},
            {"99AA999",   CountryCode::AM},
            {"9999AA9",   CountryCode::BY},
            {"AA999AA",   CountryCode::GE},
            {"AA999AAA",  CountryCode::GE},
            {"9999AAA",   CountryCode::KG},
            {"9999AA",    CountryCode::KG},
            {"999AA",     CountryCode::KG},
            {"A9999AA",   CountryCode::KG},
            {"A9999A",    CountryCode::KG},
            {"9999AA",    CountryCode::KZ},
            {"AAA999",    CountryCode::UZ},
            {"AA9999",    CountryCode::UZ},
            {"99A999999", CountryCode::UZ},
            {"9999AA99",  CountryCode::UZ},
            {"A99999999", CountryCode::UZ},
            {"99999A9A",  CountryCode::KG},
            {"99999999",  CountryCode::KG},
            {"AAA99999",  CountryCode::RU},
            {"AAA999A",   CountryCode::KG},
            {"AA9999AA",  CountryCode::KG},
            {"999A999",   CountryCode::AZ},
            {"999AA999",  CountryCode::AZ},
            {"999AAA999", CountryCode::AZ},
            {"99AAAA99",  CountryCode::AZ},
            {"999.99AAA", CountryCode::KZ},
    };

    const char STANDARD_DIGIT = '9';
    const char STANDARD_ALPHA = 'A';

    std::string standardizeLicensePlate(const std::string &plateLabel);

public:
    std::string processSquareLicensePlate(const std::string &topPlateLabel, const std::string &bottomPlateLabel);

    std::string getCountryCode(const std::string &plateLabel);
};
