#include "utils/matrix_utils.h"
#include "gf2p/gf2p.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace Eigen;

std::vector<std::pair<int, int>> getNonZeroPositions(const MatrixXi& mat) {
    std::vector<std::pair<int, int>> positions;
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            if (mat(i, j) != 0) {
                positions.emplace_back(i, j);
            }
        }
    }
    return positions;
}

bool verifyNonbinaryOrthogonality(
    const MatrixXi& H_gamma,
    const MatrixXi& H_delta,
    const GF2p& gf) {
    
    if (H_gamma.cols() != H_delta.cols() || H_gamma.rows() != H_delta.rows()) {
        return false;
    }
    
    MatrixXi H_delta_T = H_delta.transpose();
    
    for (int i = 0; i < H_gamma.rows(); ++i) {
        for (int j = 0; j < H_delta_T.cols(); ++j) {
            int sum = 0;
            for (int k = 0; k < H_gamma.cols(); ++k) {
                int term = gf.mul(H_gamma(i, k), H_delta_T(k, j));
                sum = gf.add(sum, term);
            }
            if (sum != 0) {
                return false;
            }
        }
    }
    return true;
}

void printNonbinaryMatrix(
    const MatrixXi& mat,
    const std::string& name,
    const GF2p& gf,
    int P) {
    
    std::cout << "\n=== " << name << " (" << mat.rows() << "x" << mat.cols() 
              << ", GF(2^" << gf.getSize() << ")) ===" << std::endl;
    
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << gf.toString(mat(i, j)) << " ";
            if ((j + 1) % P == 0 && j < mat.cols() -1) std::cout << "| ";
        }
        std::cout << std::endl;
        if ((i + 1) % P == 0 && i < mat.rows() -1) {
            for(int k=0; k < mat.cols() + (mat.cols()/P -1)*2; ++k) std::cout << "-";
            std::cout << std::endl;
        }
    }
}

bool writeUbsAlistNonBinary(
    const Eigen::MatrixXi& H_values,
    int gfOrder,
    const std::string& filepath) {

    const int M = H_values.rows();
    const int N = H_values.cols();

    // 统计列/行度数
    std::vector<int> dv(N, 0);
    std::vector<int> dc(M, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (H_values(i, j) != 0) {
                dc[i]++;
                dv[j]++;
            }
        }
    }

    const int cmax = dv.empty() ? 0 : *std::max_element(dv.begin(), dv.end());
    const int rmax = dc.empty() ? 0 : *std::max_element(dc.begin(), dc.end());

    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }

    // Header: N M GF
    out << N << " " << M << " " << gfOrder << "\n";
    // Max column/row weight
    out << cmax << " " << rmax << "\n";

    // Column degrees
    for (int j = 0; j < N; ++j) {
        out << dv[j] << (j + 1 < N ? " " : "\n");
    }
    // Row degrees
    for (int i = 0; i < M; ++i) {
        out << dc[i] << (i + 1 < M ? " " : "\n");
    }

    // Column-wise placeholder (cmax pairs per column, ignored by decoder)
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < cmax; ++k) {
            out << 0 << " " << 0;
            if (k + 1 < cmax) {
                out << " ";
            }
        }
        out << "\n";
    }

    // Row-wise data: each row has rmax pairs (column_index, GF_value)
    for (int i = 0; i < M; ++i) {
        int written = 0;
        for (int j = 0; j < N; ++j) {
            int val = H_values(i, j);
            if (val != 0) {
                // NB-LDPC expects 1-based column indices
                out << (j + 1) << " " << val;
                written++;
                if (written < rmax) {
                    out << " ";
                }
            }
        }
        // Fill remaining slots with zeros
        for (int k = written; k < rmax; ++k) {
            out << 0 << " " << 0;
            if (k + 1 < rmax) {
                out << " ";
            }
        }
        out << "\n";
    }

    out.close();
    return true;
}

void printAsHexLog(
    const Eigen::MatrixXi& mat,
    const std::string& name,
    const GF2p& gf,
    int P) {

    std::cout << "\n=== " << name << " (Hex Log Representation) ===" << std::endl;

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            int element = mat(i, j);

            if (element == 0) {
                // 对于零元素，打印一个点来表示空白
                std::cout << " "; 
            } else {
                // 对于非零元素：
                // 1. 计算其在 GF(2^p) 中的对数
                int log_val = gf.log(element);

                // 2. 将对数值转换为十六进制字符串
                std::stringstream ss;
                ss << std::hex << log_val;
                std::cout << ss.str();
            }
            
            // 为了对齐，在每个元素后加一个空格
            std::cout << " ";

            // 在 PxP 块之间添加垂直分隔符
            if ((j + 1) % P == 0 && j < mat.cols() - 1) {
                std::cout << "| ";
            }
        }
        std::cout << std::endl;

        // 在 J/2 组行之后添加水平分隔符
        if ((i + 1) % P == 0 && i < mat.rows() - 1) {
            // 计算分隔线的长度以实现对齐
            for(int k=0; k < (mat.cols() * 2) + (mat.cols() / P - 1) * 2 -1; ++k) {
                std::cout << "-";
            }
            std::cout << std::endl;
        }
    }
}