// MIT License
// Thanks to Cyrille Rossant for the initial python version

#include <array>
#include <iostream>

#include "xtensor-io/ximage.hpp"

#include "xtl/xvariant.hpp"

#include "xtensor/xfixed.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xeval.hpp"

#include "xtensor-blas/xlinalg.hpp"

using namespace xt;

using vector_t = xtensor_fixed<double, xshape<3>>;
using color_t = xtensor_fixed<double, xshape<3>>;

template <class E>
auto normalize(const xexpression<E>& x)
{
    auto& xd = x.derived_cast();
    return eval(xd / xt::linalg::norm(xd));
}

namespace xc
{
    // template <class T = double>
    constexpr static double inf = std::numeric_limits<double>::infinity();
}

template <class EC, class ED, class EP, class EN>
double intersect_plane(xexpression<EC>& O, xexpression<ED>& D, xexpression<EP>& P, xexpression<EN>& N)
{
    double denom = xt::linalg::dot(D, N)();  // automatically convert to double ... 
    if (std::abs(denom) < 1e-6)
    {
        return xc::inf;
    }
    double d = (xt::linalg::dot(P.derived_cast() - O.derived_cast(), N) / denom)();
    if (d < 0)
    {
        return xc::inf;
    }
    return d;
}

template <class EC, class ED, class ES>
double intersect_sphere(xexpression<EC>& O, xexpression<ED>& D, xexpression<ES>& S, double R)
{
    double a = xt::linalg::dot(D, D)();
    auto OS = O.derived_cast() - S.derived_cast();
    double b = 2 * xt::linalg::dot(D, OS)();
    double c = xt::linalg::dot(OS, OS)() - R * R;
    double disc = b * b - 4 * a * c;
    if (disc > 0)
    {
        auto distSqrt = std::sqrt(disc);
        auto q = b < 0 ? (-b - distSqrt) / 2.0 : (-b + distSqrt) / 2.0;
        double t0 = q / a;
        double t1 = c / q;
        std::tie(t0, t1) = std::make_pair(std::min(t0, t1), std::max(t0, t1));
        if (t1 >= 0)
        {
            return t0 < 0 ? t1 : t0;
        }
    }
    return xc::inf;
}

enum object_type
{
    plane, sphere
};

template <class T>
class object
{
public:

    object(const T& pos, const color_t& col)
        : position(pos), color(col)
    {
    }

    object_type type;
    vector_t position;
    color_t color;

    double diffuse = -1;
    double specular = -1;
};

template <class T>
class xsphere : public object<T>
{
public:

    xsphere(const T& pos, double rad, const T& col)
        : object<T>(pos, col), radius(rad)
    {
    }

    double radius;

    template <class E>
    auto get_color(xexpression<E>& /*pos*/)
    {
        return object<T>::color;
    }
};

template <class T>
class xplane : public object<T>
{
public:
    T normal;

    xplane(const T& pos, const T& nrm, const T& col)
        : object<T>(pos, col), normal(nrm)
    {
    }

    template <class E>
    auto get_color(xexpression<E>& pos)
    {
        auto& posd = pos.derived_cast();
        if (int(posd(0) * 2) % 2 == int(posd[2] * 2) % 2)
        {
            return object<T>::color;
        }
        else
        {
            return color_t(object<T>::color * 2);
        }
    }
};

template <class EO, class ED, class T>
double intersect(xexpression<EO>& O, xexpression<ED>& D, xsphere<T>& obj)
{
    return intersect_sphere(O, D, obj.position, obj.radius);
}

template <class EO, class ED, class T>
double intersect(xexpression<EO>& O, xexpression<ED>& D, xplane<T>& obj)
{
    return intersect_plane(O, D, obj.position, obj.normal);
}

template <class T, class EM>
vector_t get_normal(xsphere<T>& obj, xexpression<EM>& M)
{
    return normalize(M.derived_cast() - obj.position);
}

template <class T, class EM>
vector_t get_normal(xplane<T>& obj, xexpression<EM>& M)
{
    return obj.normal;
}

static vector_t O({0., 0.35, -1.}); // Camera

template <class T>
using obj_variant = xtl::variant<xsphere<T>, xplane<T>>;

template <class T>
using scene_t = std::vector<obj_variant<T>>;

static constexpr double ambient = .05;
static constexpr double diffuse_c = 1.;
static constexpr double specular_c = 1.;
static constexpr double specular_k = 50;

static color_t color_light({1, 1, 0.5});

template <class T, class EO, class ED>
std::tuple<ptrdiff_t, vector_t, vector_t, color_t>
trace_ray(scene_t<T>& scene, xexpression<EO>& rayO, xexpression<ED>& rayD)
{
    static vector_t L({5, 5, -10});     // Light

    double t = xc::inf;
    std::size_t obj_idx;
    for (std::size_t i = 0; i < scene.size(); ++i)
    {
        auto& obj = scene[i];
        auto t_obj = xtl::visit([&rayO, &rayD](auto& obj)
        {
            return intersect(rayO, rayD, obj);
        }, obj);

        if (t_obj < t)
        {
            t = t_obj;
            obj_idx = i;
        }
    }

    if (t == xc::inf)
    {
        return std::make_tuple(-1, vector_t{}, vector_t{}, color_t{});
    }
    auto& obj = scene[obj_idx];
    vector_t M = eval(rayO.derived_cast() + rayD.derived_cast() * t);
    vector_t N = xtl::visit([&M](auto& obj) { return get_normal(obj, M); }, obj);
    color_t color = xtl::visit([&M](auto& obj) { return obj.get_color(M); }, obj);
    auto toL = normalize(L - M);
    auto toO = normalize(O - M);
    for (std::size_t i = 0; i < scene.size(); ++i)
    {
        // find if shadowed
        if (i == obj_idx) continue;

        auto& obj_sh = scene[i];

        vector_t xM = M + N * 0.0001;
        vector_t xtoL = toL;

        double l = xtl::visit([&xM, &xtoL](auto& obj) {
            return intersect(xM, xtoL, obj);
        }, obj_sh);

        if (l < xc::inf)
        {
            return std::make_tuple(-1, vector_t{}, vector_t{}, color_t{});
        }
    }
    color_t col_ray = color_t(ambient);
    // Diffuse shading
    col_ray += xtl::visit([](const auto& obj) { return obj.diffuse != -1 ? obj.diffuse : diffuse_c; }, obj) * std::max(xt::linalg::dot(N, toL)(), 0.) * color;
    // Blinn-Phong shading (specular).
    col_ray += xtl::visit([](const auto& obj) { return obj.specular != -1 ? obj.specular : specular_c; }, obj) * std::pow(std::max(xt::linalg::dot(N, normalize(toL + toO))(), 0.0), specular_k) * color_light; 
    return std::make_tuple(ptrdiff_t(obj_idx), M, N, col_ray);

}

int main()
{
    auto scene = scene_t<vector_t>();
    scene.push_back(xsphere<vector_t>({{.75, .1, 1}}, 0.6, {{0.7, 0, 0}}));
    auto gp = xplane<vector_t>({{0., -.5, 0.}}, {{0, 1, 0}}, {{0.3, 0.3, 0.3}});
    gp.specular = 0.2;
    gp.diffuse = 0.8;
    scene.push_back(gp);
    auto sp2 = xsphere<vector_t>({{-0.5, .1, 1}}, 0.3, {{0, 0.6, 0.8}});
    sp2.specular = 0.1;
    sp2.diffuse = 2;
    scene.push_back(sp2);

    std::cout << "Raytracing" << std::endl;

    static constexpr int w = 400;
    static constexpr int h = 300;

    double r = float(w) / h;
    // # Screen coordinates: x0, y0, x1, y1.
    std::array<double, 4> S({-1., -1. / r + .25, 1., 1. / r + .25});

    auto px_x = xt::linspace(S[0], S[2], w);
    auto px_y = xt::linspace(S[1], S[3], h);

    xtensor<unsigned char, 3> img = xtensor<unsigned char, 3>::from_shape({h, w, 3});

    const int depth_max = 5;

    vector_t Q({0., 0., 0.});     // Camera pointing to.
    vector_t O({0., 0.35, -1.});  // Camera

    for (std::size_t i = 0; i < w; ++i)
    {
        if (i % 10 == 0)
        {
            std::cout << i / float(w) * 100 << "%" << std::endl;
        }
        for (std::size_t j = 0; j < h; ++j)
        {
            auto x = px_x(i);
            auto y = px_y(j);
            color_t col = xt::zeros<double>({3});
            Q(0) = x;
            Q(1) = y;
            auto D = normalize(Q - O);
            double depth = 0;
            auto rayO = O;
            auto rayD = D;
            double reflection = 1.;

            while (depth < depth_max)
            {
                auto traced = trace_ray(scene, rayO, rayD);
                if (std::get<0>(traced) == -1)
                    break;
                rayO = std::get<1>(traced) + std::get<2>(traced) * 0.0001;
                rayD = normalize(rayD - 2 * xt::linalg::dot(rayD, std::get<2>(traced)) * std::get<2>(traced));
                col += reflection * std::get<3>(traced);
                reflection *= 0.5;
                ++depth;
            }

            col = xt::clip(col, 0, 1) * 255;
            img(h - j - 1, i, 0) = col(0);
            img(h - j - 1, i, 1) = col(1);
            img(h - j - 1, i, 2) = col(2);
        }
    }

    xt::dump_image("test.png", img);
    return 0;
}