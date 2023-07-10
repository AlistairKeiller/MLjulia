using MLDatasets

mutable struct Model
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
    activations::Vector{Function}
    activations′::Vector{Function}
end

function Model(layers::Vector{Tuple{Int,Int,Function,Function}})
    Model([(2 * rand(out, in) .- 1) / sqrt(in) for (in, out, a, a′) ∈ layers], [(2 * rand(out) .- 1) / sqrt(in) for (in, out, a, a′) in layers], [a for (in, out, a, a′) in layers], [a′ for (in, out, a, a′) in layers])
end

function error(model::Model, in::Vector{Float32}, out::Vector{Float32})
    for (w, b, a) ∈ zip(model.weights, model.biases, model.activations)
        in = a(w * in + b)
    end
    sum((in - out) .^ 2)
end

function train(model::Model, train_data, learning_rate=1e-3, batch_size=256, β1=0.9, β2=0.999, ϵ=1e-8)
    # Initialize first and second moment estimates for weights and biases
    m_w = [zeros(size(w)) for w in model.weights]
    v_w = [zeros(size(w)) for w in model.weights]
    m_b = [zeros(size(b)) for b in model.biases]
    v_b = [zeros(size(b)) for b in model.biases]
    t = 0

    while true
        t += 1
        println("error: ", sum(error(model, in, out) for (in, out) ∈ test_data) / length(test_data))

        for batch in [train_data[i:min(i+batch_size-1, end)] for i in 1:batch_size:length(train_data)]
            for (x, y) in batch
                # Forward pass
                pre_activations = []
                activations = [x]
                for (w, b, a) in zip(model.weights, model.biases, model.activations)
                    push!(pre_activations, w * activations[end] + b)
                    push!(activations, a(pre_activations[end]))
                end

                # Backward pass
                δ = 2 * (activations[end] - y) .* model.activations′[end](pre_activations[end])
                for l in length(model.weights):-1:1
                    ∇b = δ
                    ∇w = δ * activations[l]'
                    if l != 1
                        δ = (model.weights[l]' * δ) .* model.activations′[l](pre_activations[l-1])
                    end

                    # Update weights
                    m_w[l] = β1 * m_w[l] + (1 - β1) * ∇w
                    v_w[l] = β2 * v_w[l] + (1 - β2) * ∇w .^ 2
                    m_hat_w = m_w[l] / (1 - β1^t)
                    v_hat_w = v_w[l] / (1 - β2^t)
                    model.weights[l] -= learning_rate .* m_hat_w ./ (sqrt.(v_hat_w) .+ ϵ)

                    # Update biases
                    m_b[l] = β1 * m_b[l] + (1 - β1) * ∇b
                    v_b[l] = β2 * v_b[l] + (1 - β2) * ∇b .^ 2
                    m_hat_b = m_b[l] / (1 - β1^t)
                    v_hat_b = v_b[l] / (1 - β2^t)
                    model.biases[l] -= learning_rate .* m_hat_b ./ (sqrt.(v_hat_b) .+ ϵ)
                end
            end
        end
    end
end

function relu(x)
    max.(x, 0)
end

function relu′(x)
    x .> 0
end

layers::Vector{Tuple{Int,Int,Function,Function}} = [(28 * 28, 128, relu, relu′), (128, 10, relu, relu′)]
model = Model(layers)
train_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:train)]
test_data = [(vec(in), [Float32(i == out) for i ∈ 0:9]) for (in, out) ∈ MNIST(:test)]

train(model, train_data, learning_rate)