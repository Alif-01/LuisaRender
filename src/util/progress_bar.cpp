//
// Created by Mike Smith on 2022/4/7.
//

#include <iostream>

#include <luisa/core/stl.h>
#include <util/progress_bar.h>

namespace luisa::render {

ProgressBar::ProgressBar(bool silent, uint32_t width) noexcept
    : _progress{0.0f}, _width{width}, _clock{}, _silent{silent} {}

void ProgressBar::reset() noexcept {
    _clock.tic();
    _progress = 0.0f;
}

void ProgressBar::done() noexcept {
    update(1.0);
    if (_silent) return;
    std::cout << std::endl;
}

void ProgressBar::update(double progress) noexcept {
    using namespace std::chrono_literals;
    _progress = std::clamp(std::max(_progress, progress), 0.0, 1.0);
    if (_silent) return;
    auto pos = static_cast<uint32_t>(_width * _progress);
    auto dt = static_cast<double>(_clock.toc()) * 1e-3;
    std::cout << "\33[2K\r[";
    for (auto i = 0; i < _width; ++i) {
        if (i < pos) {
            std::cout << complete_char;
        } else if (i == pos) {
            std::cout << heading_char;
        } else {
            std::cout << incomplete_char;
        }
    }
    if (_progress != 0.0 && _progress != 1.0) [[likely]] {
        auto prompt = luisa::format(
            "({:.4f}s | {:.1f}% | ETA {:.4f}s)",
            dt, _progress * 100, (1.f - _progress) / _progress * dt);
        std::cout << "] " << prompt;
    } else {
        auto prompt = luisa::format(
            "({:.4f}s | {:.1f}%)", dt, _progress * 100);
        std::cout << "] " << prompt;
    }
    std::cout.flush();
}

}// namespace luisa::render
