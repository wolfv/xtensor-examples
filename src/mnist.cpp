#include <xtensor/xtensor.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>

#include <xtensor-blas/xlinalg.hpp>

#include <mnist/mnist_reader.hpp>

#define MNIST_SIZE 784

namespace activations
{
    template<class T>
    inline xt::xarray<T> sigmoid(const xt::xarray<T>& x)
    {
        return T(1.0) / (T(1.0) + xt::exp(-x));
    };

    template<class T>
    inline xt::xarray<T> tanh(const xt::xarray<T>& x)
    {
        return xt::tanh(x);
    };
}

template <class T>
class neural_net
{
public:

    neural_net(std::size_t input, std::size_t hidden, std::size_t out, T learningrate);

    void train(const xt::xarray<T>& inputs,const xt::xarray<T>& targets);

    xt::xarray<T> query(const xt::xarray<T>& inputs) const;

private:

    xt::xarray<T> m_wih;
    xt::xarray<T> m_who;

    std::size_t m_inputs;
    std::size_t m_hiddens;
    std::size_t m_outputs;

    T m_learningRate;
    std::function<xt::xarray<T>(xt::xarray<T>)> m_activation;
};

template <class T>
neural_net<T>::neural_net(std::size_t inputs, std::size_t hiddens, std::size_t outs, T lr) :
    m_inputs(inputs), m_hiddens(hiddens), m_outputs(outs), m_learningRate(lr)
{
    m_wih = xt::random::randn({m_hiddens, m_inputs}, 0.0, std::pow(m_inputs , -0.5));
    m_who = xt::random::randn({m_outputs, m_hiddens}, 0.0, std::pow(m_hiddens, -0.5));

    m_activation = activations::sigmoid<T>;
}

template <class T>
void neural_net<T>::train(const xt::xarray<T>& inputs,const xt::xarray<T>& targets)
{
    xt::xarray<T> hiddenInputs = xt::linalg::dot(m_wih, inputs);
    xt::xarray<T> hiddenOutputs = m_activation(hiddenInputs);

    xt::xarray<T> finalInputs = xt::linalg::dot(m_who, hiddenOutputs);
    xt::xarray<T> finalOutputs = m_activation(finalInputs);

    xt::xarray<T> outputErrors = targets - finalOutputs;

    xt::xarray<T> tmp_t = xt::transpose(m_who);
    xt::xarray<T> hiddenErrors = xt::linalg::dot(tmp_t, outputErrors);

    tmp_t = xt::transpose(hiddenOutputs);
    m_who += m_learningRate * xt::linalg::dot((outputErrors * finalOutputs * (1.0f - finalOutputs)),
        tmp_t);

    tmp_t = xt::transpose(inputs);
    m_wih += m_learningRate * xt::linalg::dot((hiddenErrors * hiddenOutputs * (1.0f - hiddenOutputs)),
        tmp_t);
}

template <class T>
xt::xarray<T> neural_net<T>::query(const xt::xarray<T>& inputs) const
{
    xt::xarray<T> hiddenInputs = xt::linalg::dot(m_wih, inputs);
    xt::xarray<T> hiddenOutputs = m_activation(hiddenInputs);

    xt::xarray<T> finalInputs = xt::linalg::dot(m_who, hiddenOutputs);
    xt::xarray<T> finalOutputs = m_activation(finalInputs);

    return finalOutputs;
}

template <class T, class D>
void check(neural_net<T>& net, D& dataset)
{
    std::vector<int> correct;
    int jdx = 0;
    for (const auto& img : dataset.test_images)
    {
        uint8_t lbl = dataset.test_labels[jdx];

        xt::xarray<uint8_t> uimg;
        std::array<std::size_t, 2> shape = {MNIST_SIZE, 1};
        auto arr = xt::adapt(img, shape);
        auto ftensor = xt::cast<float>(arr);
        auto normalized = ftensor / 255.0 * 0.99 + 0.01;

        auto result = net.query(normalized);

        int max = 0;
        for (int i = 0; i < 10; ++i)        
        {
            if (result(i, 0) > result(max, 0))
            {
                max = i;
            }
        }
        
        max == lbl ? correct.push_back(1) : correct.push_back(0);
        ++jdx;
    }

    int sum = std::accumulate(correct.begin(), correct.end(), 0);
    double acc = (double) sum / correct.size();

    std::cout << "Recognition accuracy is: " << std::setprecision(5)  << acc << std::endl;
}

template <class T, class D>
void train(neural_net<T>& net, D& dataset)
{
    int idx = 0;
    const int total = 60000;

    for (const auto& img : dataset.training_images)
    {
        auto& lbl = dataset.training_labels[idx];
        
        xt::xarray<uint8_t> uimg;
        std::array<std::size_t, 2> shape = {MNIST_SIZE, 1};
        auto arr = xt::adapt(img, shape);

        auto ftensor = xt::cast<float>(arr);
        auto normalized = ftensor / 255.0 * 0.99 + 0.01;
    
        xt::xtensor<float, 2> targets = xt::zeros<float>({10, 1}) + 0.01;
        targets(lbl, 0) = 0.99;

        net.train(normalized, targets);

        if (idx % 1000 == 0)
        {
            std::cout << (double(idx) / double(total)) * 100 << "%" << std::endl;
            check(net, dataset);
        }
        ++idx;
    }
}


int main(int argc,char** argv)
{
    neural_net<float> net(MNIST_SIZE, 100, 10, 0.2);
    auto dataset = mnist::read_dataset();
    train(net, dataset);
    return 0;
}