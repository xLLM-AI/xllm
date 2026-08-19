/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "time_predictor.h"

#include <glog/logging.h>

#include <cmath>
#include <vector>

namespace xllm {

TimePredictor::TimePredictor(bool if_profile_prefix, bool is_prefill)
    : if_profile_prefix_(if_profile_prefix), is_prefill_(is_prefill) {
  if (is_prefill) {
    if (!if_profile_prefix) {
      coefficients_ = Eigen::VectorXd::Zero(3);
    } else {
      coefficients_ = Eigen::VectorXd::Zero(5);
    }
  } else {
    coefficients_ = Eigen::VectorXd::Zero(3);
  }
}

void TimePredictor::fit_for_decode(
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        time_profiling_data) {
  // construct Vandermonde matrix
  trained_ = true;
  int32_t m = time_profiling_data.size();
  int32_t n = 3;
  Eigen::MatrixXd matrix(m, n);
  for (int32_t i = 0; i < m; ++i) {
    int32_t token_length = std::get<0>(time_profiling_data[i]);
    int32_t seq_num = std::get<1>(time_profiling_data[i]);

    matrix(i, 0) = 1.0;  // the index 0 is always for constant
    matrix(i, 1) = seq_num;
    matrix(i, 2) = seq_num * (token_length - 1);
  }
  // construct target vector
  Eigen::VectorXd target(m);
  for (int32_t i = 0; i < m; ++i) {
    target(i) = std::get<2>(time_profiling_data[i]);
  }

  // get coefficients
  coefficients_ = matrix.colPivHouseholderQr().solve(target);

  // Calculate predictions and errors
  double sum_abs_error = 0.0;
  double sum_percentage_error = 0.0;

  for (const auto& data : time_profiling_data) {
    int32_t token_length = std::get<0>(data);
    int32_t seq_num = std::get<1>(data);
    double actual = std::get<2>(data);

    double prediction = coefficients_(0) + coefficients_(1) * seq_num +
                        coefficients_(2) * seq_num * (token_length - 1);

    double abs_error = std::abs(prediction - actual);
    sum_abs_error += abs_error;

    sum_percentage_error += abs_error / actual;
  }

  // Calculate MAE and MAPE
  double mae = sum_abs_error / time_profiling_data.size();
  double mape = (sum_percentage_error / time_profiling_data.size()) * 100.0;

  // output equation
  LOG(INFO) << "Fitted equation: time = " << coefficients_(1) << " * seq_num + "
            << coefficients_(2) << " * seq_num * prefix_length + "
            << coefficients_(0);
  LOG(INFO) << "MAE: " << mae << ", MAPE: " << mape << "%";

  check_coefficients_non_neg(n);
}

void TimePredictor::fit_for_prefill(
    const std::vector<std::pair<int32_t, double>>& time_profiling_data) {
  // construct Vandermonde matrix
  trained_ = true;
  int32_t m = time_profiling_data.size();
  int32_t n = 3;  // use 2-degree polynomial
  Eigen::MatrixXd matrix(m, n);
  for (int32_t i = 0; i < m; ++i) {
    for (int32_t j = 0; j < n; ++j) {
      matrix(i, j) = std::pow(time_profiling_data[i].first, j);
    }
  }
  // construct target vector
  Eigen::VectorXd target(m);
  for (int32_t i = 0; i < m; ++i) {
    target(i) = time_profiling_data[i].second;
  }
  // get coefficients
  coefficients_ = matrix.colPivHouseholderQr().solve(target);

  // Calculate predictions and errors
  double sum_abs_error = 0.0;
  double sum_percentage_error = 0.0;

  for (int32_t i = 0; i < m; ++i) {
    double prediction = 0.0;
    double x = time_profiling_data[i].first;

    // Calculate prediction using the fitted polynomial
    for (int32_t j = 0; j < coefficients_.size(); ++j) {
      prediction += coefficients_(j) * std::pow(x, j);
    }

    double actual = time_profiling_data[i].second;
    double abs_error = std::abs(prediction - actual);
    sum_abs_error += abs_error;
    sum_percentage_error += abs_error / actual;
  }

  // Calculate MAE and MAPE
  double mae = sum_abs_error / m;
  double mape = (sum_percentage_error / m) * 100.0;

  // Output equation
  std::stringstream equation;
  equation << "Fitted equation: time = ";
  for (int32_t i = 0; i < coefficients_.size(); ++i) {
    if (i > 0) equation << " + ";
    equation << coefficients_(i);
    if (i == 1)
      equation << " * x";
    else if (i > 1)
      equation << " * x^" << i;
  }
  LOG(INFO) << equation.str();
  LOG(INFO) << "MAE: " << mae << ", MAPE: " << mape << "%";

  check_coefficients_non_neg(n);
}

std::vector<double> TimePredictor::get_coefficients() {
  std::vector<double> coeffs;
  for (int32_t i = 0; i < coefficients_.size(); ++i) {
    coeffs.push_back(coefficients_(i));
  }
  return coeffs;
}

double TimePredictor::get_constant_overhead() {
  double result = coefficients_(0);
  if (result < 0) {
    LOG(ERROR) << "Negative constant term: " << result;
    result = 0.0;
  }
  return result;
}

int32_t TimePredictor::get_quadratic_root(int32_t prefix_length,
                                          double budget) {
  CHECK(is_prefill_) << "This function is only for prefill.";
  double a = 0.0, b = 0.0, c = 0.0;
  if (if_profile_prefix_) {
    a = coefficients_(1);
    b = coefficients_(2) + coefficients_(3) * prefix_length;
    c = coefficients_(4) * prefix_length - budget;
  } else {
    a = coefficients_(1);
    b = coefficients_(2);
    c = -budget;
  }
  double discriminant = b * b - 4 * a * c;
  if (discriminant < 0) {
    LOG(ERROR) << "No real roots exist for the given budget: " << budget;
    return prefix_length;  // No real roots exist
  } else {
    double root = (-b + std::sqrt(discriminant)) / (2 * a);
    if (root >= 0) {
      return static_cast<int32_t>(root) + prefix_length;
    } else {
      LOG(ERROR) << "roots are negative for the given budget: " << budget;
      return prefix_length;  // Both roots are negative
    }
  }
}

void TimePredictor::fit_for_prefill(
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        time_profiling_data) {
  trained_ = true;
  int32_t m = time_profiling_data.size();
  int32_t n = 5;
  Eigen::MatrixXd matrix(m, n);

  for (int32_t i = 0; i < m; ++i) {
    int32_t token_length = std::get<0>(time_profiling_data[i]);
    int32_t prefix_length = std::get<1>(time_profiling_data[i]);
    int32_t diff = token_length - prefix_length;

    matrix(i, 0) = 1.0;          // the index 0 is always for constant
    matrix(i, 1) = diff * diff;  // (token_length-prefix_length)^2
    matrix(i, 2) = diff;         // (token_length-prefix_length)
    matrix(i, 3) =
        diff * prefix_length;      // (token_length-prefix_length)*prefix_length
    matrix(i, 4) = prefix_length;  // prefix_length
  }
  // construct target vector
  Eigen::VectorXd target(m);
  for (int32_t i = 0; i < m; ++i) {
    target(i) = std::get<2>(time_profiling_data[i]);
  }
  // get coefficients
  coefficients_ = matrix.colPivHouseholderQr().solve(target);

  // Calculate errors
  double sum_abs_error = 0.0;
  double sum_percentage_error = 0.0;

  for (const auto& data : time_profiling_data) {
    int32_t token_length = std::get<0>(data);
    int32_t prefix_length = std::get<1>(data);
    int32_t actual = std::get<2>(data);
    int32_t diff = token_length - prefix_length;

    double prediction = coefficients_(0) + coefficients_(1) * (diff * diff) +
                        coefficients_(2) * diff +
                        coefficients_(3) * (diff * prefix_length) +
                        coefficients_(4) * prefix_length;

    double abs_error = std::abs(prediction - actual);
    sum_abs_error += abs_error;
    sum_percentage_error += abs_error / actual;
  }

  double mae = sum_abs_error / time_profiling_data.size();
  double mape = (sum_percentage_error / time_profiling_data.size()) * 100.0;

  // output equation
  LOG(INFO) << "Fitted equation: time = " << coefficients_(1) << " * diff^2 + "
            << coefficients_(2) << " * diff + " << coefficients_(3)
            << " * (diff * prefix_length) + " << coefficients_(4)
            << " * prefix_length + " << coefficients_(0);
  LOG(INFO) << "MAE: " << mae << ", MAPE: " << mape << "%";

  check_coefficients_non_neg(n);
}

void TimePredictor::check_coefficients_non_neg(int32_t num) {
  for (int32_t i = 0; i < num; ++i) {
    if (coefficients_(i) < 0) {
      LOG(ERROR) << "Negative coefficient: " << coefficients_(i)
                 << ", set it to 0.";
      coefficients_(i) = 0;
    }
  }
}

double TimePredictor::predict_time(int32_t length,
                                   int32_t prefix_length,
                                   bool if_need_add_constant_term) {
  double result = 0.0;
  if (if_need_add_constant_term) {
    result = coefficients_(0);
  }
  if (is_prefill_) {
    if (!if_profile_prefix_) {
      // use prefix-free profile
      int32_t effective_length = length - prefix_length;
      double power = effective_length;
      for (int32_t i = 1; i < coefficients_.size(); ++i) {
        result += coefficients_(i) * power;
        power *= effective_length;
      }

    } else {
      // prefix profile
      int32_t diff = length - prefix_length;
      result += (coefficients_(1) * diff * diff + coefficients_(2) * diff +
                 coefficients_(3) * diff * prefix_length +
                 coefficients_(4) * prefix_length);
    }
  } else {
    result += (coefficients_(1) * 1 + coefficients_(2) * (length - 1));
  }
  if (result < 0) {
    LOG(ERROR) << "Negative time prediction: " << result
               << ". Input param: length:" << length
               << " prefix_length:" << prefix_length;
    result = 0;
  }
  return result;
}

}  // namespace xllm
