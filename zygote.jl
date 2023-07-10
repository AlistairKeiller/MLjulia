using Zygote, MLDatasets

mutable struct Model
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
    activations::Vector{Function}
end

function Model(layers::Vector{Tuple{Int, Int, Function}})
    Model([(2*rand(out, in).-1)/sqrt(in) for (in, out, a) ∈ layers], [(2*rand(out).-1)/sqrt(in) for (in, out, a) in layers], [a for (in, out, a) in layers])
end

function error(model::Model, in::Vector{Float32}, out::Vector{Float32})
    for (w, b, a) ∈ zip(model.weights, model.biases, model.activations)
        in = a(w * in + b)
    end
    sum((in-out).^2)
end

function train(model, train_data, test_data, learning_rate)
    while true
        println("error: ", sum(error(model, in, out) for (in, out) ∈ test_data)/length(test_data))
        for (in, out) ∈ train_data
            g = gradient(Params([model.weights, model.biases])) do
                error(model, in, out)
            end
            model.weights -= learning_rate*g[model.weights]
            model.biases -= learning_rate*g[model.biases]
        end
    end
end

function relu(x)
    max.(x,0)
end

layers::Vector{Tuple{Int, Int, Function}} = [(28*28, 128, relu), (128, 10, relu)]
model = Model(layers)
train_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:train)]
test_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:test)]
learning_rate = Float32(1e-3)

train(model, train_data, test_data, learning_rate)