#ifndef BF590DAD_E7A6_487A_A775_5127D9045258
#define BF590DAD_E7A6_487A_A775_5127D9045258

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVRow
{
    public:
        CSVRow(const char delim = ','):m_delim(delim)
        {}
        std::string const& operator[](std::size_t index) const
        {
            return m_data[index];
        }
        std::size_t size() const
        {
            return m_data.size();
        }
        std::vector<std::string> fields() const
        {
            return m_data;
        }
        void readNextRow(std::istream& str)
        {
            std::string         line;
            std::getline(str, line);

            std::stringstream   lineStream(line);
            std::string         cell;

            m_data.clear();
            while(std::getline(lineStream, cell, m_delim))
            {
                m_data.push_back(cell);
            }
            // This checks for a trailing comma with no data after it.
            if (!lineStream && cell.empty())
            {
                // If there was a trailing comma then add an empty element.
                m_data.push_back("");
            }
        }
    private:
        std::vector<std::string>    m_data;
        char                        m_delim;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}   
// int main()
// {
//     std::ifstream       file("plop.csv");

//     CSVRow              row;
//     while(file >> row)
//     {
//         std::cout << "4th Element(" << row[3] << ")\n";
//     }
// }
#endif /* BF590DAD_E7A6_487A_A775_5127D9045258 */
