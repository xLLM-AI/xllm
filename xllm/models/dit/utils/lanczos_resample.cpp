/**
 * Lanczos Resample – standalone C++ implementation.
 *
 * Ported from Pillow (PIL) src/libImaging/Resample.c.
 * Uses the classic two-pass separable resampling: horizontal then vertical.
 *
 * Original code copyright:
 *   Copyright (c) 1997-2005 by Secret Labs AB
 *   Copyright (c) 1995-2005 by Fredrik Lundh
 */

#include "lanczos_resample.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace lanczos {
namespace {

inline double sinc_filter(double x) {
  if (x == 0.0) {
    return 1.0;
  }
  x = x * M_PI;
  return std::sin(x) / x;
}

inline double lanczos_filter(double x) {
  if (-3.0 <= x && x < 3.0) {
    return sinc_filter(x) * sinc_filter(x / 3.0);
  }
  return 0.0;
}

constexpr double kLanczosSupport = 3.0;

constexpr int32_t kPrecisionBits = 32 - 8 - 2;  // = 22

uint8_t clip8_lut[1280];
uint8_t* clip8_ptr = nullptr;

void init_clip8_lut() {
  if (clip8_ptr) return;
  for (int32_t i = 0; i < 1280; i++) {
    int32_t val = i - 640;
    clip8_lut[i] = static_cast<uint8_t>(val < 0 ? 0 : (val > 255 ? 255 : val));
  }
  clip8_ptr = &clip8_lut[640];
}

inline uint8_t clip8(int32_t in) { return clip8_ptr[in >> kPrecisionBits]; }

int32_t precompute_coeffs(int32_t in_size,
                          double in0,
                          double in1,
                          int32_t out_size,
                          std::vector<int32_t>& bounds,
                          std::vector<double>& kk) {
  // Prepare for horizontal/vertical stretch.
  double scale = (in1 - in0) / out_size;
  double filterscale = scale;
  if (filterscale < 1.0) {
    filterscale = 1.0;
  }

  // Determine support size (length of resampling filter).
  double support = kLanczosSupport * filterscale;

  // Maximum number of coefficients per output pixel.
  int32_t ksize = static_cast<int32_t>(std::ceil(support)) * 2 + 1;

  // Coefficient buffer.
  kk.resize(static_cast<size_t>(out_size) * ksize, 0.0);
  bounds.resize(static_cast<size_t>(out_size) * 2);

  for (int32_t xx = 0; xx < out_size; xx++) {
    double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    double ss = 1.0 / filterscale;

    // Round the value.
    int32_t xmin = static_cast<int32_t>(center - support + 0.5);
    if (xmin < 0) xmin = 0;

    // Round the value.
    int32_t xmax = static_cast<int32_t>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;

    double* k = &kk[static_cast<size_t>(xx) * ksize];
    for (int32_t x = 0; x < xmax; x++) {
      double w = lanczos_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int32_t x = 0; x < xmax; x++) {
      if (ww != 0.0) {
        k[x] /= ww;
      }
    }
    // Remaining values should stay empty if they are used despite of xmax.
    for (int32_t x = xmax; x < ksize; x++) {
      k[x] = 0.0;
    }
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return ksize;
}

void normalize_coeffs_8bpc(int32_t out_size,
                           int32_t ksize,
                           std::vector<double>& prekk,
                           std::vector<int32_t>& ikk) {
  ikk.resize(prekk.size());
  for (size_t x = 0; x < static_cast<size_t>(out_size) * ksize; x++) {
    if (prekk[x] < 0) {
      ikk[x] = static_cast<int32_t>(-0.5 + prekk[x] * (1 << kPrecisionBits));
    } else {
      ikk[x] = static_cast<int32_t>(0.5 + prekk[x] * (1 << kPrecisionBits));
    }
  }
}

void resample_horizontal_8bpc(uint8_t* dst,
                              int32_t dst_w,
                              int32_t dst_h,
                              const uint8_t* src,
                              int32_t src_w,
                              int32_t /*src_h*/,
                              int32_t channels,
                              int32_t y_offset,
                              int32_t ksize,
                              const int32_t* bounds,
                              const int32_t* kk) {
  for (int32_t yy = 0; yy < dst_h; yy++) {
    for (int32_t xx = 0; xx < dst_w; xx++) {
      int32_t xmin = bounds[xx * 2 + 0];
      int32_t xmax = bounds[xx * 2 + 1];  // count, not end
      const int32_t* k = &kk[xx * ksize];

      for (int32_t c = 0; c < channels; c++) {
        int32_t ss = 1 << (kPrecisionBits - 1);  // 0.5 for rounding
        for (int32_t x = 0; x < xmax; x++) {
          ss += static_cast<int32_t>(src[(yy + y_offset) * src_w * channels +
                                         (x + xmin) * channels + c]) *
                k[x];
        }
        dst[yy * dst_w * channels + xx * channels + c] = clip8(ss);
      }
    }
  }
}

void resample_vertical_8bpc(uint8_t* dst,
                            int32_t dst_w,
                            int32_t dst_h,
                            const uint8_t* src,
                            int32_t src_w,
                            int32_t /*src_h*/,
                            int32_t channels,
                            int32_t ksize,
                            const int32_t* bounds,
                            const int32_t* kk) {
  for (int32_t yy = 0; yy < dst_h; yy++) {
    int32_t ymin = bounds[yy * 2 + 0];
    int32_t ymax = bounds[yy * 2 + 1];  // count
    const int32_t* k = &kk[yy * ksize];

    for (int32_t xx = 0; xx < dst_w; xx++) {
      for (int32_t c = 0; c < channels; c++) {
        int32_t ss = 1 << (kPrecisionBits - 1);
        for (int32_t y = 0; y < ymax; y++) {
          ss += static_cast<int32_t>(
                    src[(y + ymin) * src_w * channels + xx * channels + c]) *
                k[y];
        }
        dst[yy * dst_w * channels + xx * channels + c] = clip8(ss);
      }
    }
  }
}

void resample_horizontal_f32(float* dst,
                             int32_t dst_w,
                             int32_t dst_h,
                             const float* src,
                             int32_t src_w,
                             int32_t /*src_h*/,
                             int32_t channels,
                             int32_t y_offset,
                             int32_t ksize,
                             const int32_t* bounds,
                             const double* kk) {
  for (int32_t yy = 0; yy < dst_h; yy++) {
    for (int32_t xx = 0; xx < dst_w; xx++) {
      int32_t xmin = bounds[xx * 2 + 0];
      int32_t xmax = bounds[xx * 2 + 1];
      const double* k = &kk[xx * ksize];

      for (int32_t c = 0; c < channels; c++) {
        double ss = 0.0;
        for (int32_t x = 0; x < xmax; x++) {
          ss += static_cast<double>(src[(yy + y_offset) * src_w * channels +
                                        (x + xmin) * channels + c]) *
                k[x];
        }
        dst[yy * dst_w * channels + xx * channels + c] = static_cast<float>(ss);
      }
    }
  }
}

void resample_vertical_f32(float* dst,
                           int32_t dst_w,
                           int32_t dst_h,
                           const float* src,
                           int32_t src_w,
                           int32_t /*src_h*/,
                           int32_t channels,
                           int32_t ksize,
                           const int32_t* bounds,
                           const double* kk) {
  for (int32_t yy = 0; yy < dst_h; yy++) {
    int32_t ymin = bounds[yy * 2 + 0];
    int32_t ymax = bounds[yy * 2 + 1];
    const double* k = &kk[yy * ksize];

    for (int32_t xx = 0; xx < dst_w; xx++) {
      for (int32_t c = 0; c < channels; c++) {
        double ss = 0.0;
        for (int32_t y = 0; y < ymax; y++) {
          ss += static_cast<double>(
                    src[(y + ymin) * src_w * channels + xx * channels + c]) *
                k[y];
        }
        dst[yy * dst_w * channels + xx * channels + c] = static_cast<float>(ss);
      }
    }
  }
}

}  // namespace

void resize_8bpc(const uint8_t* src,
                 int32_t src_w,
                 int32_t src_h,
                 int32_t channels,
                 int32_t dst_w,
                 int32_t dst_h,
                 uint8_t* dst) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || channels <= 0 ||
      channels > 4) {
    throw std::invalid_argument(
        "lanczos::resize_8bpc: invalid dimensions or channels");
  }

  init_clip8_lut();

  // Pre-compute horizontal coefficients.
  std::vector<int32_t> bounds_h;
  std::vector<double> kk_h;
  int32_t ksize_h = precompute_coeffs(
      src_w, 0.0, static_cast<double>(src_w), dst_w, bounds_h, kk_h);

  // Pre-compute vertical coefficients.
  std::vector<int32_t> bounds_v;
  std::vector<double> kk_v;
  int32_t ksize_v = precompute_coeffs(
      src_h, 0.0, static_cast<double>(src_h), dst_h, bounds_v, kk_v);

  // First used row in the source image.
  int32_t ybox_first = bounds_v[0];
  // Last used row in the source image.
  int32_t ybox_last = bounds_v[static_cast<size_t>(dst_h) * 2 - 2] +
                      bounds_v[static_cast<size_t>(dst_h) * 2 - 1];

  bool need_h = (dst_w != src_w);
  bool need_v = (dst_h != src_h);

  // Convert to fixed-point coefficients.
  std::vector<int32_t> ikk_h, ikk_v;
  normalize_coeffs_8bpc(dst_w, ksize_h, kk_h, ikk_h);
  normalize_coeffs_8bpc(dst_h, ksize_v, kk_v, ikk_v);

  if (need_h && need_v) {
    // Two-pass: horizontal first, temp buffer sized dst_w * (ybox_last -
    // ybox_first).
    int32_t temp_h = ybox_last - ybox_first;
    std::vector<uint8_t> temp(static_cast<size_t>(dst_w) * temp_h * channels);

    resample_horizontal_8bpc(temp.data(),
                             dst_w,
                             temp_h,
                             src,
                             src_w,
                             src_h,
                             channels,
                             ybox_first,
                             ksize_h,
                             bounds_h.data(),
                             ikk_h.data());

    // Shift bounds for vertical pass.
    for (int32_t i = 0; i < dst_h; i++) {
      bounds_v[i * 2] -= ybox_first;
    }

    resample_vertical_8bpc(dst,
                           dst_w,
                           dst_h,
                           temp.data(),
                           dst_w,
                           temp_h,
                           channels,
                           ksize_v,
                           bounds_v.data(),
                           ikk_v.data());
  } else if (need_h) {
    resample_horizontal_8bpc(dst,
                             dst_w,
                             src_h,
                             src,
                             src_w,
                             src_h,
                             channels,
                             0,
                             ksize_h,
                             bounds_h.data(),
                             ikk_h.data());
  } else if (need_v) {
    resample_vertical_8bpc(dst,
                           src_w,
                           dst_h,
                           src,
                           src_w,
                           src_h,
                           channels,
                           ksize_v,
                           bounds_v.data(),
                           ikk_v.data());
  } else {
    // No scaling needed, just copy.
    size_t total = static_cast<size_t>(src_w) * src_h * channels;
    std::memcpy(dst, src, total);
  }
}

void resize_f32(const float* src,
                int32_t src_w,
                int32_t src_h,
                int32_t channels,
                int32_t dst_w,
                int32_t dst_h,
                float* dst) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || channels <= 0) {
    throw std::invalid_argument(
        "lanczos::resize_f32: invalid dimensions or channels");
  }

  // Pre-compute horizontal coefficients.
  std::vector<int32_t> bounds_h;
  std::vector<double> kk_h;
  int32_t ksize_h = precompute_coeffs(
      src_w, 0.0, static_cast<double>(src_w), dst_w, bounds_h, kk_h);

  // Pre-compute vertical coefficients.
  std::vector<int32_t> bounds_v;
  std::vector<double> kk_v;
  int32_t ksize_v = precompute_coeffs(
      src_h, 0.0, static_cast<double>(src_h), dst_h, bounds_v, kk_v);

  // First / last used row in the source image.
  int32_t ybox_first = bounds_v[0];
  int32_t ybox_last = bounds_v[static_cast<size_t>(dst_h) * 2 - 2] +
                      bounds_v[static_cast<size_t>(dst_h) * 2 - 1];

  bool need_h = (dst_w != src_w);
  bool need_v = (dst_h != src_h);

  if (need_h && need_v) {
    int32_t temp_h = ybox_last - ybox_first;
    std::vector<float> temp(static_cast<size_t>(dst_w) * temp_h * channels);

    resample_horizontal_f32(temp.data(),
                            dst_w,
                            temp_h,
                            src,
                            src_w,
                            src_h,
                            channels,
                            ybox_first,
                            ksize_h,
                            bounds_h.data(),
                            kk_h.data());

    for (int32_t i = 0; i < dst_h; i++) {
      bounds_v[i * 2] -= ybox_first;
    }

    resample_vertical_f32(dst,
                          dst_w,
                          dst_h,
                          temp.data(),
                          dst_w,
                          temp_h,
                          channels,
                          ksize_v,
                          bounds_v.data(),
                          kk_v.data());
  } else if (need_h) {
    resample_horizontal_f32(dst,
                            dst_w,
                            src_h,
                            src,
                            src_w,
                            src_h,
                            channels,
                            0,
                            ksize_h,
                            bounds_h.data(),
                            kk_h.data());
  } else if (need_v) {
    resample_vertical_f32(dst,
                          src_w,
                          dst_h,
                          src,
                          src_w,
                          src_h,
                          channels,
                          ksize_v,
                          bounds_v.data(),
                          kk_v.data());
  } else {
    // No scaling needed, just copy.
    size_t total = static_cast<size_t>(src_w) * src_h * channels;
    std::memcpy(dst, src, total * sizeof(float));
  }
}

}  // namespace lanczos