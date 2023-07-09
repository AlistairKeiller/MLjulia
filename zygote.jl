using Zygote, MLDatasets, Random

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

function train(model, train_data, test_data, batch_size, learning_rate)
    while true
        println("error: ", sum(error(model, in, out) for (in, out) ∈ test_data)/length(test_data))
        
        # Shuffle the training data at the beginning of each epoch
        Random.shuffle!(train_data)
        
        # Split the training data into batches
        for i in 1:batch_size:length(train_data)
            batch = train_data[i:min(i+batch_size-1, end)]
            
            g = gradient(Params([model.weights, model.biases])) do
                sum(error(model, in, out) for (in, out) in batch)/length(batch)
            end
            
            model.weights -= learning_rate*g[model.weights]
            model.biases -= learning_rate*g[model.biases]
        end
    end
end

function relu(x)
    max.(x,0)
end

function softmax(x)
    exp.(x)/sum(exp.(x))
end
layers::Vector{Tuple{Int, Int, Function}} = [(28*28, 128, relu), (128, 10, relu)]
model = Model(layers)
train_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:train)]
test_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:test)]
batch_size = 256
learning_rate = Float32(1e-3)

train(model, train_data, test_data, batch_size, learning_rate)