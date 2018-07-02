#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xindex_view.hpp>

// def initialize_centroids(points, k):
//     """returns k centroids from the initial points"""
//     centroids = points.copy()
//     np.random.shuffle(centroids)
//     return centroids[:k]

using namespace xt;

template <class E>
xarray<double> initialize_centroids(xexpression<E>& points, int k)
{
    auto centroids = points.derived_cast();
    random::shuffle(centroids);
    return view(centroids, range(0, k));
}

template <class E, class F>
xarray<std::size_t> closest_centroid(xexpression<E>& points, xexpression<F>& centroids)
{
    auto& d_points = points.derived_cast();
    auto& d_centroids = centroids.derived_cast();
    auto distances = sqrt(sum(pow(d_points - view(d_centroids, all(), newaxis()), 2), { 2 }));
    return argmin(distances, 0);
}

template <class E, class F, class G>
auto move_centroids(xexpression<E>& points, xexpression<F>& closest, xexpression<G>& centroids)
{
    auto new_centroids = empty_like(centroids);
    auto& d_centroids = centroids.derived_cast();

    // auto size = e.size();
    // auto s = sum(std::forward<E>(e), std::forward<X>(axes));
    // return std::move(s) / static_cast<double>(size / s.size());

    for (std::size_t i = 0; i < d_centroids.shape()[0]; ++i)
    {
        auto w = where(equal(closest.derived_cast(), i));
        std::vector<std::size_t> wv(w.size());
        for (std::size_t j = 0; j < w.size(); ++j)
        {
            wv[j] = w[j][0];
        }
        auto fv = view(points.derived_cast(), islice(std::move(wv)));
        std::size_t sz = fv.size();
        if (sz != 0)
        {
            auto sm = sum(fv, {0});
            view(new_centroids, i) = sm / (sz / sm.size());
        }
    }
    return new_centroids;
}

// def move_centroids(points, closest, centroids):
//     """returns the new centroids assigned from the points closest to them"""
//     return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


int main(int argc, char const *argv[])
{
    if (argc > 1)
    {
        xt::random::seed(std::stoi(argv[1]));
    }

    xarray<double> points = concatenate(xtuple(
        random::randn<double>({150, 2}) * 0.75 + xarray<double>{1, 0},
        random::randn<double>({50, 2}) * 0.25 + xarray<double>{-0.5, 0.5},
        random::randn<double>({50, 2}) * 0.5 + xarray<double>{-0.5, 0.5}
    ), 0);

    auto centroids = initialize_centroids(points, 3);
    for (std::size_t i = 0; i < 100; ++i)
    {
        std::cout << "ITER: " << i << std::endl;
        std::cout << centroids << std::endl;
        auto cc = closest_centroid(points, centroids);
        centroids = move_centroids(points, cc, centroids);
    }

    return 0;
}