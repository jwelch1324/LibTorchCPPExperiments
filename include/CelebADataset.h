#ifndef AE952F75_35FF_4291_8340_BC231E7AEF3B
#define AE952F75_35FF_4291_8340_BC231E7AEF3B

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <string>
#include <experimental/filesystem>
#include <map>

namespace fs = std::experimental::filesystem;

#include "CSVRow.h"

namespace torch{
    namespace data{
        namespace datasets{
            class CelebA : public Dataset<CelebA>
            {
            public:
                enum class Mode { kTrain, kValid, kTest, kAll };
                enum class Target { kAttr, kIdentity, kBBox, kLandmarks };

                CelebA(
                    const std::string& root,
                    Mode mode = Mode::kTrain,
                    std::vector<Target> target = std::vector<Target>({Target::kAttr})
                )
                {
                    //std::ifstream splits_file(fs::path(root) / "list_eval_partition.txt");
                    //std::ifstream identity_file();
                    auto identity_data = loadLabelFile(fs::path(root) / "identity_CelebA.txt");
                }

                Example<> get(size_t index) override
                {
                    //return {{0},{0}};Then 
                }

                std::map<std::string,std::vector<std::string>> loadLabelFile(fs::path file)
                {
                    std::map<std::string,std::vector<std::string>> data;
                    std::ifstream label_file(file);
                    CSVRow row;
                    while(label_file >> row)
                    {
                        data[row[0]] = std::vector<std::string>(row.fields().begin()+1,row.fields().end());
                    }
                    return data;
                }

            };
        }
    }
}

#endif /* AE952F75_35FF_4291_8340_BC231E7AEF3B */
