using MLDatasets

mutable struct Model
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
    activations::Vector{Function}
    activations′::Vector{Function}
end

function Model(layers::Vector{Tuple{Int, Int, Function, Function}})
    Model([(2*rand(out, in).-1)/sqrt(in) for (in, out, a, a′) ∈ layers], [(2*rand(out).-1)/sqrt(out) for (in, out, a, a′) in layers], [a for (in, out, a, a′) in layers], [a′ for (in, out, a, a′) in layers])
end

function error(model::Model, in::Vector{Float32}, out::Vector{Float32})
    for (w, b, a) ∈ zip(model.weights, model.biases, model.activations)
        in = a(w * in + b)
    end
    sum((in-out).^2)
end

function train(model::Model, train_data, learning_rate)
    while true
        println("error: ", sum(error(model, in, out) for (in, out) ∈ test_data)/length(test_data))
        for (x, y) in train_data
            # Forward pass
            pre_activations = []
            activations = [x]
            for (w, b, a) in zip(model.weights, model.biases, model.activations)
                push!(pre_activations, w * activations[end] + b)
                push!(activations, a(pre_activations[end]))
            end

            # Backward pass
            δ = 2*(activations[end] - y) .* model.activations′[end](pre_activations[end])
            for l in length(model.weights):-1:1
                ∇b = δ
                ∇w = δ * activations[l]'
                if l != 1
                    δ = (model.weights[l]' * δ) .* model.activations′[l](pre_activations[l-1])
                end

                # Update weights and biases
                model.weights[l] -= learning_rate * ∇w
                model.biases[l] -= learning_rate * ∇b
            end
        end
    end
end

function relu(x)
    max.(x,0)
end

function relu′(x)
    x .> 0
end

layers::Vector{Tuple{Int, Int, Function, Function}} = [(28*28, 128, relu, relu′), (128, 10, relu, relu′)]
model = Model(layers)
train_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:train)]
test_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:test)]
learning_rate = Float32(1e-3)

train(model, train_data, learning_rate)