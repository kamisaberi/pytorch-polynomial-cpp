#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>

using namespace std;

struct Polynomial3 : torch::nn::Module {
    torch::Tensor a, b, c, d;

    Polynomial3() : a(torch::randn(1)),
                    b(torch::randn(1)),
                    c(torch::randn(1)),
                    d(torch::randn(1)) {
        register_parameter("a", a, true);
        register_parameter("b", b, true);
        register_parameter("c", c, true);
        register_parameter("d", d, true);
    }

    torch::Tensor forward(torch::Tensor x) const {
        return a + b * x + c * torch::pow(x, 2) + d * torch::pow(x, 3);
    }

    std::string toString() const {
        return a.toString() + " + " + b.toString() + "x + " + c.toString() + "x^2 + " + d.toString() + "x^3";
    }
};

int main() {
    const torch::Tensor x = torch::linspace(-std::numbers::pi, std::numbers::pi, 2000);
    const torch::Tensor y = torch::sin(x);
    const auto model = Polynomial3();
    // cout << model << endl;
    // torch::Tensor y_hat = model.forward(x);

    // std::cout << y << std::endl;
    // std::cout << y_hat << std::endl;

    for (const auto& param : model.parameters()) {
      cout << param << endl;
    }

    auto criterion = torch::nn::MSELoss();
    auto optimizer = torch::optim::SGD(model.parameters(), 1e-6);
    for (int i = 1; i < 20000; i++) {
        auto y_hat = model.forward(x);
        auto loss = criterion(y, y_hat);
        if (i % 100 == 99)
            cout << i << " " << loss.item() << std::endl;;
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    cout << model.toString();

    return 0;
}
