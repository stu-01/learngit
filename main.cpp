#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// 定义HSV颜色范围的结构体
struct ColorRange {
    cv::Scalar lower;
    cv::Scalar upper;
    std::string name;
    cv::Scalar drawColor; // 绘制时用的BGR颜色
};

// 获取对应颜色的掩膜并绘制轮廓和标签
void detectColor(const cv::Mat& hsv, cv::Mat& output, const ColorRange& colorRange) {
    cv::Mat mask;
    cv::inRange(hsv, colorRange.lower, colorRange.upper, mask);

    // 可选：平滑去噪
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);


    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > 500) { // 忽略小噪声
            cv::Rect rect = cv::boundingRect(contours[i]);
            cv::rectangle(output, rect, colorRange.drawColor, 2);
            cv::putText(output, colorRange.name, cv::Point(rect.x, rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, colorRange.drawColor, 2);
        }
    }
}

int main(int argc, char** argv) {
    // 检查输入参数
    if (argc < 2) {
        std::cout << "请提供视频文件路径，如：./color_detect video.mp4" << std::endl;
        return -1;
    }

    // 打开视频
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cout << "无法打开视频文件：" << argv[1] << std::endl;
        return -1;
    }

    // 定义颜色范围（HSV）
    std::vector<ColorRange> colorRanges = {
        { cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), "Red",   cv::Scalar(0, 0, 255) },
        { cv::Scalar(100, 150, 70), cv::Scalar(130, 255, 255), "Blue",  cv::Scalar(255, 0, 0) },
        { cv::Scalar(40, 70, 70),  cv::Scalar(80, 255, 255), "Green", cv::Scalar(0, 255, 0) }
    };

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Mat result = frame.clone();

        for (const auto& colorRange : colorRanges) {
            detectColor(hsv, result, colorRange);
        }

        cv::imshow("Color Detection", result);
        if (cv::waitKey(30) == 27) break; // 按 ESC 退出
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
