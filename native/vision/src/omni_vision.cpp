// omni_vision: C++17 OpenCV preprocessing for the ViT image model.
//
// decode (JPEG/PNG bytes) or take a raw HxWx3 uint8 BGR frame
//   -> resize the short side to `resize_short` (bilinear)
//   -> center-crop `crop` x `crop`
//   -> BGR to RGB, scale to [0, 1], normalize per channel (ImageNet mean/std)
//   -> float32 NCHW
//
// Zero-copy at both ends: raw frames are wrapped as cv::Mat views of the NumPy
// buffer, and results are written straight into a heap buffer that the
// returned NumPy array owns through a capsule. The GIL is released while
// decoding and transforming, so Python threads can preprocess concurrently.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct Params {
  int resize_short = 256;
  int crop = 224;
  std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
  std::array<float, 3> std{0.229f, 0.224f, 0.225f};
};

Params make_params(int resize_short, int crop, const std::vector<float>& mean,
                   const std::vector<float>& stdv) {
  if (crop <= 0 || resize_short < crop) {
    throw std::invalid_argument("need 0 < crop <= resize_short");
  }
  if (mean.size() != 3 || stdv.size() != 3) {
    throw std::invalid_argument("mean and std must have 3 values (R, G, B)");
  }
  Params p;
  p.resize_short = resize_short;
  p.crop = crop;
  for (int c = 0; c < 3; ++c) {
    if (stdv[c] <= 0.f) throw std::invalid_argument("std values must be positive");
    p.mean[c] = mean[c];
    p.std[c] = stdv[c];
  }
  return p;
}

// Writes one 3 x crop x crop image into `out` (CHW, RGB, normalized).
void transform(const cv::Mat& bgr, const Params& p, float* out) {
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    throw std::invalid_argument("expected a non-empty 8-bit 3-channel image");
  }
  const int h = bgr.rows, w = bgr.cols;
  const double scale = static_cast<double>(p.resize_short) / std::min(h, w);
  const int nh = std::max(p.crop, static_cast<int>(std::lround(h * scale)));
  const int nw = std::max(p.crop, static_cast<int>(std::lround(w * scale)));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  const int y0 = (nh - p.crop) / 2, x0 = (nw - p.crop) / 2;
  const cv::Mat roi = resized(cv::Rect(x0, y0, p.crop, p.crop));

  // Fused BGR->RGB swap, scale, normalize and HWC->CHW transpose: one pass.
  const size_t plane = static_cast<size_t>(p.crop) * p.crop;
  float* r = out;
  float* g = out + plane;
  float* b = out + 2 * plane;
  const float sr = 1.f / (255.f * p.std[0]), sg = 1.f / (255.f * p.std[1]),
              sb = 1.f / (255.f * p.std[2]);
  const float or_ = p.mean[0] / p.std[0], og = p.mean[1] / p.std[1], ob = p.mean[2] / p.std[2];
  for (int y = 0; y < p.crop; ++y) {
    const uint8_t* row = roi.ptr<uint8_t>(y);
    const size_t base = static_cast<size_t>(y) * p.crop;
    for (int x = 0; x < p.crop; ++x) {
      const uint8_t* px = row + 3 * x;  // B, G, R
      r[base + x] = px[2] * sr - or_;
      g[base + x] = px[1] * sg - og;
      b[base + x] = px[0] * sb - ob;
    }
  }
}

cv::Mat decode(const char* data, size_t len) {
  cv::Mat raw(1, static_cast<int>(len), CV_8UC1, const_cast<char*>(data));
  cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
  if (img.empty()) throw std::invalid_argument("could not decode image bytes");
  return img;
}

// A float32 NCHW array that owns `buf` (no copy into NumPy).
py::array_t<float> owning_array(std::unique_ptr<float[]> buf, size_t n, int crop) {
  float* data = buf.release();
  py::capsule owner(data, [](void* p) { delete[] static_cast<float*>(p); });
  const py::ssize_t c = crop;
  return py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t{3}, c, c},
                            {3 * c * c * py::ssize_t(sizeof(float)), c * c * py::ssize_t(sizeof(float)),
                             c * py::ssize_t(sizeof(float)), py::ssize_t(sizeof(float))},
                            data, owner);
}

py::array_t<float> preprocess_encoded(py::bytes data, int resize_short, int crop,
                                      std::vector<float> mean, std::vector<float> stdv) {
  const Params p = make_params(resize_short, crop, mean, stdv);
  char* ptr = nullptr;
  py::ssize_t len = 0;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &ptr, &len) != 0) {
    throw py::error_already_set();
  }
  auto buf = std::make_unique<float[]>(3 * static_cast<size_t>(crop) * crop);
  {
    py::gil_scoped_release release;
    transform(decode(ptr, static_cast<size_t>(len)), p, buf.get());
  }
  return owning_array(std::move(buf), 1, crop);
}

py::array_t<float> preprocess_batch(std::vector<py::bytes> items, int resize_short, int crop,
                                    std::vector<float> mean, std::vector<float> stdv) {
  const Params p = make_params(resize_short, crop, mean, stdv);
  std::vector<std::pair<const char*, size_t>> views;
  views.reserve(items.size());
  for (auto& it : items) {
    char* ptr = nullptr;
    py::ssize_t len = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(it.ptr(), &ptr, &len) != 0) throw py::error_already_set();
    views.emplace_back(ptr, static_cast<size_t>(len));
  }
  const size_t per = 3 * static_cast<size_t>(crop) * crop;
  auto buf = std::make_unique<float[]>(per * std::max<size_t>(views.size(), 1));
  {
    py::gil_scoped_release release;
    for (size_t i = 0; i < views.size(); ++i) {
      transform(decode(views[i].first, views[i].second), p, buf.get() + i * per);
    }
  }
  return owning_array(std::move(buf), views.size(), crop);
}

py::array_t<float> preprocess_frame(
    py::array_t<uint8_t, py::array::c_style> frame, int resize_short, int crop,
    std::vector<float> mean, std::vector<float> stdv) {
  const Params p = make_params(resize_short, crop, mean, stdv);
  py::buffer_info info = frame.request();
  if (info.ndim != 3 || info.shape[2] != 3) {
    throw std::invalid_argument("frame must be an HxWx3 uint8 BGR array");
  }
  // A view of the NumPy buffer: no copy of the input frame.
  cv::Mat view(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC3,
               info.ptr, static_cast<size_t>(info.strides[0]));
  auto buf = std::make_unique<float[]>(3 * static_cast<size_t>(crop) * crop);
  {
    py::gil_scoped_release release;
    transform(view, p, buf.get());
  }
  return owning_array(std::move(buf), 1, crop);
}

}  // namespace

PYBIND11_MODULE(omni_vision, m) {
  m.doc() = "OpenCV preprocessing for ViT: decode, resize, center-crop, normalize to NCHW float32";
  const std::vector<float> mean{0.485f, 0.456f, 0.406f}, stdv{0.229f, 0.224f, 0.225f};
  m.def("preprocess_encoded", &preprocess_encoded, py::arg("data"), py::arg("resize_short") = 256,
        py::arg("crop") = 224, py::arg("mean") = mean, py::arg("std") = stdv,
        "Decode JPEG/PNG bytes to a (1, 3, crop, crop) float32 array.");
  m.def("preprocess_batch", &preprocess_batch, py::arg("items"), py::arg("resize_short") = 256,
        py::arg("crop") = 224, py::arg("mean") = mean, py::arg("std") = stdv,
        "Decode a list of JPEG/PNG byte strings to an (N, 3, crop, crop) float32 array.");
  m.def("preprocess_frame", &preprocess_frame, py::arg("frame"), py::arg("resize_short") = 256,
        py::arg("crop") = 224, py::arg("mean") = mean, py::arg("std") = stdv,
        "Transform a raw HxWx3 uint8 BGR frame (zero-copy view) to (1, 3, crop, crop) float32.");
  m.attr("opencv_version") = CV_VERSION;
}
