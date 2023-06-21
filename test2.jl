using MLDatasets

mutable struct Layer
    weights::Matrix{Float32}
    biases::Vector{Float32}
    value::Vector{Float32}
    error::Vector{Float32}
end

mutable struct ActivationLayer
    activation::Function
    activation′::Function
    value::Vector{Float32}
    error::Vector{Float32}
end

mutable struct Model
    layers::Vector{Union{Layer, ActivationLayer}}
end

function Layer(in::Int, out::Int)
    Layer((2*rand(out, in).-1)/sqrt(in), (2*rand(out).-1)/sqrt(in), zeros(out), zeros(out))
end

function Relu(in::Int)
    ActivationLayer(x -> max.(x,0), x -> x .> 0, zeros(in), zeros(in))
end

function SoftMax(in::Int)
    ActivationLayer(x -> exp.(x)/sum(exp.(x)), x -> [exp(v) * sum(exp.(x[x .!= v])) for v ∈ x]/sum(exp.(x))^2, zeros(in), zeros(in))
end

function Sigmoid(in::Int)
    ActivationLayer(x -> 1/(1+exp.(-x)), x -> exp.(-x)/(1+exp.(-x)).^2, zeros(in), zeros(in))
end

function evaluate(model::Model, in::Vector{Float32})
    for layer in model.layers
        if typeof(layer) == Layer
            in = layer.value = layer.weights*in+layer.biases
        end
        if typeof(layer) == ActivationLayer
            in = layer.value = layer.value = layer.activation(in)
        end
    end
    in
end

function backprop(model::Model, out::Vector{Float32})
    model.layers[end].error = model.layers[end].value - out
    for i ∈ length(model.layers)-1:1
        if typeof(model.layers[i+1]) == ActivationLayer
            model.layers[i].error = model.layers[i+1].error .* model.layers[i+1].activation′(model.layers[i].value)
        end
        if typeof(model.layers[i+1]) == Layer
            model.layers[i].error = transpose(model.layers[i+1].weights) * model.layers[i+1].error
        end
    end
end

function grad(model::Model, in::Vector{Float32}, learning_rate::Float32)
    for i ∈ 1:length(model.layers)
        if typeof(model.layers[i]) == Layer
            model.layers[i].weights -= learning_rate * (model.layers[i].error .* transpose(i > 1 ? model.layers[i-1].value : in))
            model.layers[i].biases -= learning_rate * model.layers[i].error
        end
    end
end

function error(model::Model, in::Vector{Float32}, out::Vector{Float32})
    return sum((evaluate(model, in)-out).^2)/2
end

model = Model([Layer(28*28,10), SoftMax(10)])
train_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:train)]
test_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:test)]
learning_rate = Float32(1e-3)

for _ ∈ 0:10
    println("error: ", sum(error(model, in, out) for (in, out) ∈ test_data)/length(test_data))
    for (in, out) in train_data
        evaluate(model, in)
        backprop(model, out)
        grad(model, in, learning_rate)
    end
end
println(evaluate(model, train_data[1][1]))
