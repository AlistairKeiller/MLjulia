using MLDatasets

mutable struct Layer
    weights::Matrix{Float32}
    biases::Vector{Float32}
    Δweights::Matrix{Float32}
    Δbiases::Vector{Float32}
    a::Vector{Float32}
    z::Vector{Float32}
    δ::Vector{Float32}
end

function Layer(in::Int, out::Int)
    Layer(zeros(out, in), zeros(out), zeros(out, in), zeros(out), zeros(out), zeros(out), zeros(out))
end

mutable struct Model
    layers::Vector{Layer}
end

function σ(value::Float32)
    return 1/(1+exp(-value))
end

function σ′(value::Float32)
    return exp(-value)/(1+exp(-value))^2
end

function evaluate(model::Model, in::Vector{Float32})
    for layer ∈ model.layers
        layer.z = layer.weights*in+layer.biases
        in = layer.a = σ.(layer.z)
    end
    in
end

function backprop(model::Model, out::Vector{Float32})
    model.layers[end].δ = (model.layers[end].a - out) .* σ′.(model.layers[end].z)
    for i ∈ length(model.layers)-1:1
        model.layers[i].δ = (transpose(model.layers[i+1].weights) * model.layers[i+1].δ) .* σ′.(model.layers[i].z)
    end
end

function step_Δ(model::Model, in::Vector{Float32}, learning_rate::Float32, batch_size::Int)
    for i ∈ 1:length(model.layers)
        model.layers[i].Δweights -= learning_rate / batch_size * model.layers[i].δ .* transpose(i == 1 ? in : model.layers[i-1].a)
        model.layers[i].Δbiases -= learning_rate / batch_size * model.layers[i].δ
    end
end

function update_Δ(model::Model)
    for layer ∈ model.layers
        layer.weights += layer.Δweights
        fill!(layer.Δweights, 0)
        layer.biases += layer.Δbiases
        fill!(layer.Δbiases, 0)
    end
end

function error(model::Model, in::Vector{Float32}, out::Vector{Float32})
    return sum((evaluate(model, in)-out).^2)/2
end

model = Model([Layer(28*28,128), Layer(128,10)])
train_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:train)]
test_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:test)]
learning_rate = Float32(1e-3)
batch_size = 256


for _ in 0:100
    println("error: ", sum(error(model, in, out) for (in, out) ∈ test_data)/length(test_data))
    for (count,(in, out)) in enumerate(train_data)
        evaluate(model, in)
        backprop(model, out)
        step_Δ(model, in, learning_rate, batch_size)
        if count % batch_size == 0
            update_Δ(model)
        end
    end
end
