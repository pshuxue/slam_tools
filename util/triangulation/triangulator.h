#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

class Triangulator
{
private:
    const int min_inliners_number = 3;
    const float min_parallactic_angle = 1; // 最小视差角5度
    float cos_min_parallactic_angle;

    bool MutiViewTriangulate(const common::Matrix3x4fs &KTs, const common::Vector2fs &uvs,
                             const float &reproj_err, Eigen::Vector3f &pw);
    bool TriangulateKernel(const common::Matrix3x4fs &KTs, const common::Vector2fs &uvs,
                           Eigen::Vector3f &pw);
    float ReProjection(const Eigen::Vector3f &pw, const Eigen::Matrix<float, 3, 4> &KT,
                       const Eigen::Vector2f &uv);

    bool CheckParallacticAngle(const common::Vector3fs &ts, const std::vector<uchar> &inliners,
                               const Eigen::Vector3f &pw);

public:
    DEFINE_POINTER_TYPE(Triangulator);
    Triangulator();

    // 这里KTs是K*Twc, ts是Twc.topRight(3,1)
    bool TriangulateRansac(const common::Matrix3x4fs &KTs, const common::Vector2fs &uvs,
                           const common::Vector3fs &ts, const float &reproj_err, Eigen::Vector3f &pw,
                           std::vector<uchar> &inliners);
};

Triangulator::Triangulator()
{
    cos_min_parallactic_angle = cos(min_parallactic_angle * M_PI / 180);
}

bool Triangulator::TriangulateRansac(const common::Matrix3x4fs &KTs, const common::Vector2fs &uvs,
                                     const common::Vector3fs &ts, const float &reproj_err,
                                     Eigen::Vector3f &pw, std::vector<uchar> &inliners)
{
    CHECK_EQ(KTs.size(), uvs.size());
    CHECK_EQ(KTs.size(), ts.size());

    inliners.resize(KTs.size(), 1);
    if (KTs.size() < min_inliners_number)
        return false;

    if (MutiViewTriangulate(KTs, uvs, reproj_err, pw))
    {
        return CheckParallacticAngle(ts, inliners, pw);
    }

    int max_iter_number = 3;
    while (--max_iter_number >= 0)
    {
        int idx1 = rand() % ts.size();
        int idx2 = rand() % ts.size();
        while (idx2 == idx1)
            idx2 = rand() % ts.size();
        common::Matrix3x4fs hit_KTs{KTs[idx1], KTs[idx2]};
        common::Vector2fs hit_uvs{uvs[idx1], uvs[idx2]};
        MutiViewTriangulate(hit_KTs, hit_uvs, reproj_err, pw);

        int count_outliners = 0;
        common::Vector3fs hit_ts;
        for (int i = 0; i < KTs.size(); ++i)
        {
            if (ReProjection(pw, KTs[i], uvs[i]) > reproj_err)
            {
                count_outliners++;
                inliners[i] = 0;
            }
        }
        if (count_outliners == 0)
            break;

        if (CheckParallacticAngle(ts, inliners, pw))
            return true;
    }
    return false;
}

bool Triangulator::MutiViewTriangulate(const common::Matrix3x4fs &KTs, const common::Vector2fs &uvs,
                                       const float &reproj_err, Eigen::Vector3f &pw)
{
    TriangulateKernel(KTs, uvs, pw);

    for (int i = 0; i < KTs.size(); ++i)
    {
        if (ReProjection(pw, KTs[i], uvs[i]) > reproj_err)
        {
            return false;
        }
    }
    return true;
}

bool Triangulator::TriangulateKernel(const common::Matrix3x4fs &KTs, const common::Vector2fs &uvs,
                                     Eigen::Vector3f &pw)
{
    CHECK_EQ(KTs.size(), uvs.size());

    // 1.构造Apw = 0
    Eigen::Matrix<float, Eigen::Dynamic, 4> A;
    A.resize(KTs.size() * 2, 4);

    for (int i = 0; i < KTs.size(); ++i)
    {
        float u = uvs[i].x();
        float v = uvs[i].y();

        Eigen::Matrix<float, 1, 4> T1 = KTs[i].block<1, 4>(0, 0);
        Eigen::Matrix<float, 1, 4> T2 = KTs[i].block<1, 4>(1, 0);
        Eigen::Matrix<float, 1, 4> T3 = KTs[i].block<1, 4>(2, 0);
        Eigen::Matrix<float, 1, 4> single_row = u * T3 - T1;

        A.block<1, 4>(i * 2, 0) = single_row;
        single_row = v * T3 - T2;
        A.block<1, 4>(i * 2 + 1, 0) = single_row;
    }
    // 2.SVD分解ATA，求解Pw
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinV);
    Eigen::MatrixXf V = svd.matrixV();
    Eigen::Vector4f Pw = V.col(3) / V(3, 3);
    pw = Pw.head(3);
    return true;
}

float Triangulator::ReProjection(const Eigen::Vector3f &pw, const Eigen::Matrix<float, 3, 4> &KT,
                                 const Eigen::Vector2f &uv)
{
    Eigen::Vector3f pc = KT.topLeftCorner(3, 3) * pw + KT.topRightCorner(3, 1);
    Eigen::Vector2f reproj_puv = pc.hnormalized();
    return (reproj_puv - uv).norm();
}

bool Triangulator::CheckParallacticAngle(const common::Vector3fs &ts,
                                         const std::vector<uchar> &inliners,
                                         const Eigen::Vector3f &pw)
{
    CHECK_EQ(ts.size(), inliners.size());
    for (int i = 0; i < ts.size(); ++i)
    {
        if (inliners[i] == 0)
            continue;
        for (int j = i + 1; j < ts.size(); ++j)
        {
            if (inliners[j] == 0)
                continue;

            Eigen::Vector3f dir1 = (ts[i] - pw).normalized();
            Eigen::Vector3f dir2 = (ts[j] - pw).normalized();
            if (dir1.dot(dir2) < cos_min_parallactic_angle)
                return true;
        }
    }
    return false;
}
