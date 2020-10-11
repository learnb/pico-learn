-- learning lib
nn={}
-- constructor for fully-connected neural network
function nn:new()
    local this={}
    this.layers={}
    this.epoch=0
    this.error=0.0
    this.acc=0.0

    self.__index=self
    setmetatable(this,self)
    return this
end
function nn:add_layer(layer)
    add(self.layers, layer)
end
-- feed forward input through layers
-- x: n-d array
-- return: n-d array
function nn:feedforward(x)
    for layer in all(self.layers) do
        x=layer:activate(x)
    end
    return x
end
-- predicts a class
-- useful with sigmoid activation (interpret outputs as probabilities)
-- returns index of predicted classed
function nn:predict(x)
    local ff=self:feedforward(x)
    return np_argmax(ff)
end
-- backpropagation
-- uses mean squared sum loss function
-- x: input values (array)
-- y: target values (array)
-- lr: learning rate (0, 1)
function nn:backprop(x,y,lr)
    -- feedforward
    local output=self:feedforward(x)

    -- calculate errors and deltas
    for l=#self.layers,1,-1 do -- loop over layers backwards
        local layer=self.layers[l]
        if l==#self.layers then -- if this is output layer
            layer.error=np_vec_sub(y,output)
            layer.delta=np_vec_mult(layer.error,layer:apply_activation_derivative(output))
        else
            local next_layer=self.layers[l+1]
            layer.error=np_mv_dot(next_layer.weights, next_layer.delta)
            layer.delta=np_vec_mult(layer.error,layer:apply_activation_derivative(layer.last_activation))
        end
    end

    -- update weights
    for l=1,#self.layers do -- loop over layers
        local _in={}
        if (l!=1) then -- input is previous layer
            _in=self.layers[l-1].last_activation
        else -- input is x
            _in=x
        end
        -- apply update: w += delta*x*lr
        self.layers[l]:update(_in,self.layers[l].delta,lr)
    end
end
-- accuracy between predicted and true labels
function nn:accuracy(x,y_true)
    local mean=0
    for i=1,#x do -- each sample
        if (self:predict(x[i])==np_argmax(y_true[i])) then mean+=1 end
    end
    self.acc=mean/#x
    return self.acc
end
-- mean square error
-- y: input vector, 1-d array
-- returns: mse, scalar
function nn:mse(y,x)
    local e=np_vec_mean(np_vec_sq(np_vec_sub(y, self:feedforward(x))))
    self.error=e
    return e
end
-- train using sgd
-- x: input values, 2-d array. x[1] is the first sample vector
-- y: target values, 2-d array. y[1] is the first target vector
-- lr: learning rate (0,1)
-- max_epochs: maximum number of training iterations
-- returns: array of mse errors
function nn:train(x,y,lr,max_epochs)
    local results={}
    for i=1,max_epochs do -- each epoch
        self.epoch=i
        for s=1,#x do -- each sample
            self:backprop(x[s], y[s], lr)
        end
        -- check progress
        if i%10==0 then
            add(results, self:mse(y[1],x[1]))
        end
    end
    return results
end
-- train on one sample using sgd
-- x: input values, 1-d array. x is a sample vector
-- y: target values, 1-d array. y is a target vector
-- lr: learning rate (0,1)
-- max_epochs: maximum number of training iterations
function nn:train_step(x,y,lr)
    self:backprop(x, y, lr)
end

-- draw training stats
function nn:draw_stats()
    print("epoch: "..self.epoch, 15,108, 6)
    print("error: "..self.error, 15,114, 13)
    print("accuracy: "..(self.acc*100).."%", 3,120, 15)

end
layer={}
-- constructor for nn layer
function layer:new(n_input, n_neurons, activation, weights, bias)
    local this={}
    if activation != nil then this.activation=activation else
        this.activation="tanh"
    end
    if weights != nil then this.weights=weights else
        this.weights=np_mat_rand(n_input,n_neurons)
    end
    if bias != nil then this.bias=bias else 
        this.bias=np_vec_rand(n_neurons)
    end


    self.__index=self
    setmetatable(this,self)
    return this
end
-- calculates dot product of this layer
-- x is 1-d array (inputs)
-- returns vector: XW+B
function layer:activate(x)
    --print("x: "..#x)
    local res=np_vm_dot(x, self.weights) -- X dot W
    --print("xw: "..#res)
    res=np_vec_add(res, self.bias) -- add bias
    --print("xw+b: "..#res)
    self.last_activation=self:apply_activation(res) -- apply activiation fn
    --print("fn(xw+b): "..#self.last_activation)
    return self.last_activation
end
function layer:apply_activation(vec)
    if self.activation=="tanh" then
        vec=np_vec_func(vec, np_tanh)
    else -- assume "sigmoid"
        vec=np_vec_func(vec, np_sigmoid)
    end
    return vec
end
function layer:apply_activation_derivative(vec)
    if self.activation=="tanh" then
        vec=np_vec_func(vec, np_deriv_tanh)
    else -- assume "sigmoid"
        vec=np_vec_func(vec, np_deriv_sigmoid)
    end
    return vec
end
function layer:update(_input,_output,_lr)
    for i=1,#_input do
        for o=1,#_output do
            local d = _output[o]*_input[i]*_lr
            self.weights[i][o]=mid(-5, self.weights[i][o]+d, 5) -- clamp (-5, 5)
            --self.weights[i][o]+=d
        end
    end
end