#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <omp.h>

'''
g++ piv_fast.cpp -o piv_fast `pkg-config --cflags --libs opencv4`-O3 -march=native -ffast-math -fopenmp

'''

const std::string videoFile = "good_river.mp4";

constexpr float realWidth  = 20.0f;
constexpr float realLength = 10.0f;
constexpr float meanDepth  = 2.5f;
constexpr float surfaceToMeanFactor = 0.85f;

const std::vector<int>   winSizes   = {192, 96};
const std::vector<float> searchExp  = {2.0f, 1.5f};
const std::vector<float> overlaps   = {0.75f, 0.85f};

constexpr int   frameStep = 8;
constexpr float maxSpeed  = 3.5f;
constexpr float res_px_m  = 60.0f;


///////////

inline bool crossCorrelate(
    const cv::Mat& A,
    const cv::Mat& B,
    cv::Point2f& disp)
{
    static thread_local cv::Mat Af, Bf, C;

    if (cv::meanStdDev(A).second[0] < 0.01) return false;

    cv::dft(A, Af, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(B, Bf, cv::DFT_COMPLEX_OUTPUT);

    cv::mulSpectrums(Af, Bf, C, 0, true);
    cv::idft(C, C, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    double maxVal, mean, stddev;
    cv::Point maxLoc;
    cv::minMaxLoc(C, nullptr, &maxVal, nullptr, &maxLoc);
    cv::meanStdDev(C, mean, stddev);

    float psr = (maxVal - mean[0]) / (stddev[0] + 1e-6f);
    if (psr < 3.5f) return false;

    disp.x = float(maxLoc.x);
    disp.y = float(maxLoc.y);
    return true;
}


/////////

int main()
{
    cv::VideoCapture cap(videoFile);
    if (!cap.isOpened()) return -1;

    const float fps = cap.get(cv::CAP_PROP_FPS);
    const float dt  = 1.0f / fps;

    std::vector<cv::Mat> frames;
    for (int i = 0; i < 100; i++) {
        cv::Mat f, g;
        cap >> f;
        if (f.empty()) break;
        cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
        g.convertTo(g, CV_32F, 1.0 / 255.0);
        frames.push_back(g);
    }

    /* ---- MANUALLY SET ROI POINTS ---- */
    std::vector<cv::Point2f> src = { {x1,y1},{x2,y2},{x3,y3},{x4,y4} };
    std::vector<cv::Point2f> dst = {
        {0,0},{realWidth,0},{realWidth,realLength},{0,realLength}
    };

    cv::Mat H = cv::getPerspectiveTransform(src, dst);

    const int Hpx = int(realLength * res_px_m);
    const int Wpx = int(realWidth  * res_px_m);

    /* ---- PREPROCESS ---- */
    cv::Mat avg = cv::Mat::zeros(Hpx, Wpx, CV_32F);
    for (int i = 0; i < frames.size(); i += 5) {
        cv::Mat w;
        cv::warpPerspective(frames[i], w, H, {Wpx, Hpx});
        avg += w;
    }
    avg /= float(frames.size() / 5);

    std::vector<cv::Mat> proc;
    for (auto& f : frames) {
        cv::Mat w, g;
        cv::warpPerspective(f, w, H, {Wpx, Hpx});
        cv::GaussianBlur(w - avg, g, {0,0}, 3);
        proc.push_back((w - avg) - g);
    }

    /* ---- PIV LOOP ---- */
    std::vector<float> meanSpeeds;

    for (int k = 0; k + frameStep < proc.size(); k += frameStep) {

        cv::Mat& I1 = proc[k];
        cv::Mat& I2 = proc[k + frameStep];

        const int win = winSizes.back();
        const int step = int(win * (1.0f - overlaps.back()));
        const int search = int(win * searchExp.back());

        float sum = 0;
        int cnt = 0;

        #pragma omp parallel for reduction(+:sum,cnt)
        for (int y = win; y < I1.rows - win; y += step) {
            for (int x = win; x < I1.cols - win; x += step) {

                cv::Rect r1(x - win/2, y - win/2, win, win);
                cv::Rect r2(x - search/2, y - search/2, search, search);
                if (!r2.inside({0,0,I2.cols,I2.rows})) continue;

                cv::Mat A = I1(r1).clone();
                cv::Mat B = I2(r2).clone();
                A -= cv::mean(A)[0];
                B -= cv::mean(B)[0];

                cv::Point2f p;
                if (!crossCorrelate(A, B, p)) continue;

                float u = (p.x - search/2) / (res_px_m * dt * frameStep);
                float v = (p.y - search/2) / (res_px_m * dt * frameStep);
                float s = std::hypot(u, v);

                if (s > 0.05f && s < maxSpeed) {
                    sum += s;
                    cnt++;
                }
            }
        }

        if (cnt > 0) meanSpeeds.push_back(sum / cnt);
    }

    /* ---- DISCHARGE ---- */
    float Vsurf = std::accumulate(meanSpeeds.begin(), meanSpeeds.end(), 0.0f)
                  / meanSpeeds.size();

    float Q = Vsurf * surfaceToMeanFactor * realWidth * meanDepth;

    std::cout << "\n--- FINAL RESULTS ---\n";
    std::cout << "Mean Surface Velocity: " << Vsurf << " m/s\n";
    std::cout << "Discharge Q: " << Q << " m^3/s\n";
}
